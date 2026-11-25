# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias attention with short-conv K/V on Intel XPU."""

import os
from functools import partial
from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingConvState
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig

from vllm.models.inkling2.configs import Inkling2ModelConfig
from .ops.attention import gate_sink, plan_decode_attention, rel_decode_attention
from .ops.attention_prefill import rel_attention
from .ops.attention_triton import MODE_OUT_LSE
from .ops.linear import linear
from .ops.sconv import attention_prologue, shift_rel_proj
from .sconv import K_STREAM, V_STREAM, Inkling2ShortConv, conv_metadata

# Reusable <= 256 MiB BF16 scratch per (device, page shape), intentionally
# retained for process lifetime to avoid XPU allocator fragmentation across
# chunked-prefill steps.
_FP8_PREFILL_SCRATCH: dict[tuple[torch.device, int, int, int], torch.Tensor] = {}
_FP8_PREFILL_SCRATCH_MAX_BYTES = 256 * 1024 * 1024


def _fp8_prefill_scratch(
    device: torch.device,
    num_kv_heads: int,
    block_size: int,
    kv_width: int,
) -> torch.Tensor | None:
    """Return the shared BF16 prefill scratch for this page shape.

    The buffer is allocated once per ``(device, num_kv_heads, block_size,
    kv_width)`` so every layer using that page shape shares a single
    <= 256 MiB buffer. Returns ``None`` where the 2D-block-load fast path
    cannot run (``block_size % 16 != 0``) or a single page already exceeds
    the scratch budget.
    """
    if block_size % 16 or device.type == "meta":
        return None
    page_bf16_bytes = num_kv_heads * block_size * kv_width * 2
    max_pages = _FP8_PREFILL_SCRATCH_MAX_BYTES // page_bf16_bytes
    if not max_pages:
        return None
    buf_key = (device, num_kv_heads, block_size, kv_width)
    buf = _FP8_PREFILL_SCRATCH.get(buf_key)
    if buf is None:
        buf = torch.empty(
            (max_pages, num_kv_heads, block_size, kv_width),
            dtype=torch.bfloat16,
            device=device,
        )
        _FP8_PREFILL_SCRATCH[buf_key] = buf
    return buf


# INK2_HYBRID_PREFIX_SDPA=1 routes the bias-free KV prefix (keys older than
# rel_extent from the oldest query: [0, K0//page) where
# K0 = seq_len - nt - rel_extent + 1) of full-attention prefill layers
# through chunked oneDNN flash SDPA, merged with the Triton relative-bias
# tail via FP32 online softmax (LSE). Default 0 keeps the stock chunked-LSE
# Triton path for all pages.
_HYBRID_PREFIX_SDPA = os.environ.get("INK2_HYBRID_PREFIX_SDPA", "0") == "1"

# Shared <= 16 MiB BF16 K/V window scratch for hybrid prefix SDPA, retained
# for process lifetime: every layer of a page shape reuses one buffer so
# the per-window gather makes no allocations.
_SDPA_PREFILL_SCRATCH: dict[tuple[torch.device, int, int, int], torch.Tensor] = {}
_SDPA_PREFILL_SCRATCH_MAX_BYTES = 16 * 1024 * 1024
_SDPA_WIN_MAX_TOKENS = 8192


def _sdpa_prefill_scratch(
    device: torch.device,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
) -> torch.Tensor | None:
    """Return the shared BF16 K/V window scratch for hybrid prefix SDPA.

    The buffer is allocated once per ``(device, num_kv_heads, block_size,
    head_dim)`` as ``(2, num_kv_heads, win, head_dim)`` — K in ``[0]``, V in
    ``[1]`` — where ``win`` is the largest page-aligned token count inside
    the <= 16 MiB budget, capped at ``_SDPA_WIN_MAX_TOKENS``. Returns
    ``None`` where a single page already exceeds the budget.
    """
    if block_size % 16 or device.type == "meta":
        return None
    tok_bytes = 2 * num_kv_heads * head_dim * 2
    win_tok = min(_SDPA_WIN_MAX_TOKENS, _SDPA_PREFILL_SCRATCH_MAX_BYTES // tok_bytes)
    win_tok = (win_tok // block_size) * block_size
    if not win_tok:
        return None
    buf_key = (device, num_kv_heads, block_size, head_dim)
    buf = _SDPA_PREFILL_SCRATCH.get(buf_key)
    if buf is None:
        buf = torch.empty(
            (2, num_kv_heads, win_tok, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        _SDPA_PREFILL_SCRATCH[buf_key] = buf
    return buf


def _sdpa_flash(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias-free flash SDPA over one prefix window; returns ``(out, lse)``.

    ``q`` is ``(num_kv_heads, group, nt, head_dim)`` and ``k``/``v`` are
    ``(num_kv_heads, 1, win, head_dim)``: the ``group`` query heads of each
    KV head share the window's keys. Every key is strictly older than every
    query position, so ``is_causal=False``.
    """
    if q.device.type == "xpu":
        # The only SDPA-family op on the XPU build returning a real fp32
        # LSE (overrideable returns None, efficient returns zeros).
        out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0, is_causal=False, scale=softmax_scale
        )
        return out, lse
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, dropout_p=0.0, is_causal=False, scale=softmax_scale
    )
    return out, lse


def _merge_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    o_other: torch.Tensor,
    l_other: torch.Tensor,
    nt: int,
) -> None:
    """Merge ``(o_other, l_other)`` into ``(out, lse)`` in place (fp32 LSE)."""
    l0 = lse[:, :nt]
    l1 = l_other[:, :nt]
    m = torch.maximum(l0, l1)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w0 = torch.exp(l0 - m_safe)
    w1 = torch.exp(l1 - m_safe)
    ws = w0 + w1
    ws_safe = torch.where(ws == 0, torch.ones_like(ws), ws)
    lse[:, :nt] = torch.where(
        ws == 0, torch.full_like(m, float("-inf")), m_safe + torch.log(ws_safe)
    )
    out[:nt] = out[:nt] * (w0 / ws_safe).t().unsqueeze(-1) + o_other[:nt] * (
        w1 / ws_safe
    ).t().unsqueeze(-1)


def _single_prefill_seq_len(
    common_attn_metadata: CommonAttentionMetadata,
    num_decode_reqs: int,
    num_prefill_reqs: int,
) -> int | None:
    """Exact sequence length of a sole prefill request from CPU metadata.

    ``seq_lens_cpu_upper_bound`` is precise for prefill rows, so reading the
    first (and only) prefill row needs no device sync.
    """
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    if num_prefill_reqs != 1 or seq_lens_cpu is None:
        return None
    return int(seq_lens_cpu[num_decode_reqs].item())


class Inkling2AttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Record leading decode/verify requests for mixed batches."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        assert self.reorder_batch_threshold is not None
        (
            md.num_decode_reqs,
            md.num_prefill_reqs,
            md.num_decode_tokens,
            md.num_prefill_tokens,
        ) = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.reorder_batch_threshold
        )
        md.single_prefill_seq_len = _single_prefill_seq_len(
            common_attn_metadata, md.num_decode_reqs, md.num_prefill_reqs
        )
        return md


class Inkling2AttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder


def _kv_cache_layer_config(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> KVCacheLayerConfig:
    spec = AttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        ),
    )
    if sliding_window is not None:
        return SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
    return FullAttentionManager.layer_config(spec)


class Inkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """The KV cache of one extra loop pass of a looped `Inkling2Attention`
    (each pass of the loop core attends over its own keys and values)."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Read by the metadata builders of the layer's attention group.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config
        if prefix in forward_context.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer.

    Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from `x_q`, k from `x_k` and v from
    `x_v`, and use dynamic K/V convolutions and log-scaled q/r. Local layers
    given separate `x_k` / `x_v` (loop passes after the first) project them
    as full layers do.

    A loop core layer runs once per loop pass, each pass with its own KV and
    conv-state caches: `conv_states[p]` and pass `p`'s KV cache (this layer's
    for pass 0, `pass_caches[p - 1]` after). With `loop_share_cache`, every
    pass reads and writes this layer's KV cache (later tokens see the last
    pass's K/V).
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        self.num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        self.num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        # Align the standalone query-side projection to 16 bytes.
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            self.split_sizes,
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        self.max_decode_query_len = max_decode_query_len(vllm_config)
        cache_dtype = vllm_config.cache_config.cache_dtype
        supported: tuple[str, ...] = ("auto", "bfloat16", "fp8", "fp8_e4m3")
        if cache_dtype not in supported:
            raise ValueError(f"Inkling2 KV cache dtype must be one of {supported}")
        self.fp8_kv = cache_dtype in ("fp8", "fp8_e4m3")
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self
        if self.fp8_kv:
            # Reserve the shared BF16 prefill scratch during model load so
            # startup profiling counts it before KV cache sizing.
            _fp8_prefill_scratch(
                self.rel_logits_proj.device,
                self.num_kv_heads,
                vllm_config.cache_config.block_size,
                2 * self.head_dim,
            )
            if _HYBRID_PREFIX_SDPA and not is_local:
                # The hybrid prefix window scratch is reserved the same way:
                # startup sizing must include it and the runtime path makes
                # no per-window allocations.
                _sdpa_prefill_scratch(
                    self.rel_logits_proj.device,
                    self.num_kv_heads,
                    vllm_config.cache_config.block_size,
                    self.head_dim,
                )

    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window if self.is_local else None,
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        """The attention metadata key and KV cache of loop pass `pass_idx`."""
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = linear(x_q, weight)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = linear(x_q, weight[: sum(self.q_side_splits)])
        k = linear(x_k, weight[q_side:k_end])
        v = linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp8_kv:
            # Every loop pass owns byte storage containing unit-scale E4M3 KV.
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return the BF16 output of the attention block.

        Local layers read q/k/v from `x_q` (or `x_k` / `x_v` if given); full
        layers take `x_k`, `x_v` and the fp32 log-scaling `tau` for q/r.
        `pass_idx` picks the loop pass's caches.
        """
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        return linear(output, self.wo_ud.weight)

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int = 0,
    ) -> None:
        """Write the gated attention output following the decode/prefill split."""
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        attn_metadata = get_forward_context().attn_metadata
        layer_name, kv_cache = self._pass_cache(pass_idx)
        conv_state = self.conv_states[pass_idx]
        md = None
        if isinstance(attn_metadata, dict) and kv_cache.numel() > 0:
            md = cast(FlashAttentionMetadata, attn_metadata[layer_name])
        num_decodes, num_decode_tokens, decode_splits = 0, 0, 0
        if md is not None:
            num_decodes, num_decode_tokens, decode_splits = self._plan_decode(
                md, num_tokens
            )
        clipped_q = q.new_empty(q.shape)
        # Decode and prefill read the prologue's sheared relative logits.
        rel_logits = q.new_empty(
            (num_tokens + 128, self.num_heads, self.rel_extent + 256)
        )
        if md is not None:
            conv_md = conv_metadata(conv_state)
            assert conv_md is not None
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            attention_prologue(
                q,
                r.view(num_tokens, self.num_heads, self.d_rel),
                k,
                v,
                self.q_scale,
                self.r_scale,
                tau,
                self.rel_proj_shifted,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                key_cache,
                value_cache,
                md.slot_mapping,
                clipped_q,
                rel_logits,
                md.query_start_loc,
                md.seq_lens,
                clip_threshold=self.qk_clip,
                rel_threshold=self.rel_clip,
                norm_eps=self.eps,
            )
        if decode_splits:
            assert md is not None
            self._decode_attention(
                md,
                kv_cache,
                clipped_q[:num_decode_tokens],
                rel_logits,
                gate[:num_decode_tokens],
                sinks[:num_decode_tokens],
                output[:num_decode_tokens],
                decode_splits,
                num_decodes,
            )
        if num_decode_tokens < num_tokens:
            out = q.new_empty(
                (num_tokens - num_decode_tokens, self.num_heads, self.head_dim),
                dtype=torch.float32,
            )
            lse = q.new_empty((self.num_heads, out.shape[0]), dtype=torch.float32)
            if md is not None:
                self._prefill_attention(
                    md,
                    kv_cache,
                    clipped_q[num_decode_tokens:],
                    rel_logits[num_decode_tokens:],
                    out,
                    lse,
                    num_decodes,
                )
            else:
                out.zero_()
                lse.zero_()
            gate_sink(
                out,
                gate[num_decode_tokens:],
                lse.t(),
                sinks[num_decode_tokens:],
                output[num_decode_tokens:],
            )

    def _plan_decode(
        self, md: FlashAttentionMetadata, num_tokens: int
    ) -> tuple[int, int, int]:
        """Return `(num_decodes, num_decode_tokens, num_splits)` for
        `rel_decode_attention`: the whole batch, or in a mixed batch the
        leading decode/verify requests."""
        num_reqs = md.seq_lens.shape[0]
        plan = partial(
            plan_decode_attention,
            self.is_local,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rel_extent=self.rel_extent,
        )
        num_splits = plan(num_reqs=num_reqs, max_query_len=md.max_query_len)
        if num_splits:
            return num_reqs, num_tokens, num_splits
        num_decodes, num_decode_tokens = md.num_decode_reqs, md.num_decode_tokens
        if not 0 < num_decodes < num_reqs:
            return 0, 0, 0
        max_query_len = min(
            self.max_decode_query_len, num_decode_tokens - num_decodes + 1
        )
        num_splits = plan(num_reqs=num_decodes, max_query_len=max_query_len)
        if not num_splits:
            return 0, 0, 0
        return num_decodes, num_decode_tokens, num_splits

    def _decode_attention(
        self,
        md: FlashAttentionMetadata,
        kv_cache: torch.Tensor,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        output: torch.Tensor,
        num_splits: int,
        num_reqs: int,
    ) -> None:
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        mixed = num_reqs < md.seq_lens.shape[0]
        num_tokens = md.num_decode_tokens if mixed else md.num_actual_tokens
        max_query_len = md.max_query_len
        if mixed:
            max_query_len = min(self.max_decode_query_len, num_tokens - num_reqs + 1)
        rel_decode_attention(
            q,
            key_cache,
            value_cache,
            rel_logits,
            gate,
            sinks,
            output,
            block_table=md.block_table[:num_reqs],
            cache_seqlens=md.seq_lens[:num_reqs],
            cu_seqlens_q=md.query_start_loc[: num_reqs + 1],
            max_query_len=max_query_len,
            num_actual_tokens=num_tokens,
            softmax_scale=self.scaling,
            is_local=self.is_local,
            num_splits=num_splits,
        )

    def _prefill_attention(
        self,
        md: FlashAttentionMetadata,
        kv_cache: torch.Tensor,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        num_decodes: int,
    ) -> None:
        """Run relative attention for the requests after the decode batch."""
        num_decode_tokens = md.num_decode_tokens if num_decodes else 0
        nt = md.num_actual_tokens - num_decode_tokens
        cu_seqlens_q = md.query_start_loc[num_decodes:]
        if num_decodes:
            cu_seqlens_q = cu_seqlens_q - num_decode_tokens
        block_table = md.block_table[num_decodes:]
        cache_seqlens = md.seq_lens[num_decodes:]
        if (
            self.fp8_kv
            and block_table.shape[0] == 1
            and kv_cache.ndim == 4
            and kv_cache.shape[2] % 16 == 0
        ):
            page = kv_cache.shape[2]
            seq_len = getattr(md, "single_prefill_seq_len", None)
            if seq_len is None:
                seq_len = int(cache_seqlens[0].item())
            end_page = (seq_len + page - 1) // page
            if self.is_local:
                first_tok = max(0, seq_len - md.max_query_len - self.sliding_window)
                start_page = first_tok // page
            else:
                start_page = 0
            n_active = end_page - start_page
            # Materialize active FP8 pages into the shared <= 256 MiB BF16
            # scratch (reserved at model load) so attention_prefill uses
            # native 16-bit 2D block loads; above the budget, iterate in
            # chunks and merge via FP32 online FlashAttention LSE.
            buf = _fp8_prefill_scratch(
                q.device, kv_cache.shape[1], kv_cache.shape[2], kv_cache.shape[3]
            )
            max_pages = buf.shape[0] if buf is not None else 0
            if 0 < n_active and max_pages and end_page <= block_table.shape[1]:
                kv_fp8 = kv_cache.view(torch.float8_e4m3fn)

                def _run_chunk(
                    p_lo: int, p_hi: int, o_dst: torch.Tensor, l_dst: torch.Tensor
                ) -> None:
                    cnt = p_hi - p_lo
                    active = block_table[0, p_lo:p_hi].to(torch.int64)
                    kv_bf16 = buf[:cnt]
                    for s in range(0, cnt, 128):
                        e = min(s + 128, cnt)
                        kv_bf16[s:e].copy_(kv_fp8.index_select(0, active[s:e]))
                    kc, vc = kv_bf16.transpose(1, 2).split(self.head_dim, dim=-1)
                    remapped = torch.zeros(
                        (1, p_hi), dtype=torch.int32, device=q.device
                    )
                    remapped[0, p_lo:p_hi] = torch.arange(
                        cnt, dtype=torch.int32, device=q.device
                    )
                    rel_attention(
                        q[:nt],
                        kc,
                        vc,
                        o_dst,
                        l_dst,
                        rel_logits,
                        block_table=remapped,
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        max_query_len=md.max_query_len,
                        softmax_scale=self.scaling,
                        window=self.sliding_window if self.is_local else 0,
                        num_splits=1,
                        mode=MODE_OUT_LSE,
                        k_lo_tok=p_lo * page,
                        k_hi_tok=p_hi * page,
                    )

                # The bias-free prefix ([0, K0) where
                # K0 = seq_len - nt - rel_extent + 1) runs through chunked
                # oneDNN flash SDPA when INK2_HYBRID_PREFIX_SDPA=1; the tail
                # of rel_extent-biased keys stays on the Triton path. Only
                # full pages go to SDPA: the <page strays below K0 keep the
                # Triton tail's existing masking. Requires the CPU-metadata
                # seq len so this arm adds no device syncs.
                sdpa_end = 0
                if (
                    _HYBRID_PREFIX_SDPA
                    and not self.is_local
                    and self.rel_extent
                    and getattr(md, "single_prefill_seq_len", None) is not None
                ):
                    k0_tok = seq_len - nt - self.rel_extent + 1
                    if k0_tok > 0:
                        sdpa_end = min(end_page, k0_tok // page)
                kv_h = kv_cache.shape[1]
                sdpa_buf = None
                if sdpa_end and self.num_heads % kv_h == 0:
                    sdpa_buf = _sdpa_prefill_scratch(
                        q.device, kv_h, page, self.head_dim
                    )

                if sdpa_buf is not None:
                    group = self.num_heads // kv_h
                    sk, sv = sdpa_buf[0], sdpa_buf[1]
                    win_pages = sdpa_buf.shape[2] // page
                    # Batch = one KV head, heads = its query-head group.
                    q5 = q[:nt].permute(1, 0, 2).view(kv_h, group, nt, self.head_dim)
                    o_tmp = l_tmp = None
                    acc = False
                    for p_lo in range(0, sdpa_end, win_pages):
                        p_hi = min(p_lo + win_pages, sdpa_end)
                        cnt = p_hi - p_lo
                        active = block_table[0, p_lo:p_hi].to(torch.int64)
                        for s in range(0, cnt, 128):
                            e = min(s + 128, cnt)
                            sel = kv_fp8.index_select(0, active[s:e])
                            lo_tok = s * page
                            hi_tok = lo_tok + (e - s) * page
                            kv_shape = (kv_h, e - s, page, self.head_dim)
                            sk[:, lo_tok:hi_tok].view(kv_shape).copy_(
                                sel[..., : self.head_dim].transpose(0, 1)
                            )
                            sv[:, lo_tok:hi_tok].view(kv_shape).copy_(
                                sel[..., self.head_dim :].transpose(0, 1)
                            )
                        o5, l5 = _sdpa_flash(
                            q5,
                            sk[:, : cnt * page].unsqueeze(1),
                            sv[:, : cnt * page].unsqueeze(1),
                            self.scaling,
                        )
                        if not acc:
                            out[:nt].copy_(
                                o5.reshape(self.num_heads, nt, self.head_dim).permute(
                                    1, 0, 2
                                )
                            )
                            lse[:, :nt].copy_(l5.reshape(self.num_heads, nt))
                            acc = True
                        else:
                            if o_tmp is None:
                                o_tmp = torch.empty_like(out)
                                l_tmp = torch.empty_like(lse)
                            o_tmp[:nt].copy_(
                                o5.reshape(self.num_heads, nt, self.head_dim).permute(
                                    1, 0, 2
                                )
                            )
                            l_tmp[:, :nt].copy_(l5.reshape(self.num_heads, nt))
                            _merge_lse(out, lse, o_tmp, l_tmp, nt)
                    p_lo = sdpa_end
                    while p_lo < end_page:
                        p_hi = min(p_lo + max_pages, end_page)
                        if o_tmp is None:
                            o_tmp = torch.empty_like(out)
                            l_tmp = torch.empty_like(lse)
                        if not acc:
                            _run_chunk(p_lo, p_hi, out, lse)
                            acc = True
                        else:
                            _run_chunk(p_lo, p_hi, o_tmp, l_tmp)
                            _merge_lse(out, lse, o_tmp, l_tmp, nt)
                        p_lo = p_hi
                elif n_active <= max_pages:
                    _run_chunk(start_page, end_page, out, lse)
                else:
                    o_tmp = torch.empty_like(out)
                    l_tmp = torch.empty_like(lse)
                    first = True
                    for p_lo in range(start_page, end_page, max_pages):
                        p_hi = min(p_lo + max_pages, end_page)
                        if first:
                            _run_chunk(p_lo, p_hi, out, lse)
                            first = False
                        else:
                            _run_chunk(p_lo, p_hi, o_tmp, l_tmp)
                            _merge_lse(out, lse, o_tmp, l_tmp, nt)
                out[nt:].zero_()
                lse[:, nt:].zero_()
                return
            key_cache, value_cache = self._split_kv_cache(kv_cache)
        else:
            key_cache, value_cache = self._split_kv_cache(kv_cache)
        rel_attention(
            q[:nt],
            key_cache,
            value_cache,
            out,
            lse,
            rel_logits,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=md.max_query_len,
            softmax_scale=self.scaling,
            window=self.sliding_window if self.is_local else 0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        out[nt:].zero_()
        lse[:, nt:].zero_()

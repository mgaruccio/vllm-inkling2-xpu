# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for the opt-in hybrid oneDNN prefix SDPA path."""

import os
from types import SimpleNamespace

import pytest
import torch

import vllm_inkling2_xpu.attention as attention
from vllm_inkling2_xpu.attention import Inkling2Attention

# Small geometry that keeps every tensor CPU-sized: 24 heads in 8 KV-head
# groups of 3, 64-token pages, a 1024-token relative-bias window.
H, KVH, GRP, D, PAGE, REL = 24, 8, 3, 128, 64, 1024
SCALE = D**-0.5


def _fp8_kv(npages: int) -> torch.Tensor:
    """A (pages, kv_h, page, 2*D) uint8 cache holding valid E4M3 values."""
    bf16 = torch.randn(npages, KVH, PAGE, 2 * D, dtype=torch.bfloat16)
    return bf16.to(torch.float8_e4m3fn).view(torch.uint8)


def _seq_kv(kv: torch.Tensor, block_table: torch.Tensor, seq_len: int):
    """K/V as (kv_h, seq_len, D) bf16 in block-table (logical) order."""
    pages = kv.view(torch.float8_e4m3fn)[block_table.long()].to(torch.bfloat16)
    k = pages[..., :D].permute(1, 0, 2, 3).reshape(KVH, -1, D)
    v = pages[..., D:].permute(1, 0, 2, 3).reshape(KVH, -1, D)
    return k[:, :seq_len], v[:, :seq_len]


def _make_layer() -> Inkling2Attention:
    layer = object.__new__(Inkling2Attention)
    layer.fp8_kv = True
    layer.is_local = False
    layer.sliding_window = 0
    layer.rel_extent = REL
    layer.num_heads = H
    layer.num_kv_heads = KVH
    layer.head_dim = D
    layer.scaling = SCALE
    return layer


def _make_md(seq_len: int, nt: int, block_table: torch.Tensor, cpu_len=True):
    return SimpleNamespace(
        num_actual_tokens=nt,
        num_decode_tokens=0,
        max_query_len=nt,
        query_start_loc=torch.tensor([0, nt], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table=block_table.unsqueeze(0),
        single_prefill_seq_len=seq_len if cpu_len else None,
    )


def _bias_table() -> torch.Tensor:
    """Deterministic (H, rel_extent) bias for the test's fake tail."""
    return torch.linspace(-1.0, 1.0, H * REL, dtype=torch.float64).view(H, REL)


def _reference(
    q: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    seq_len: int,
    nt: int,
):
    """Full attention in fp64: causal + bias for ``dist < rel_extent``.

    Returns ``(out, lse)`` shaped ``(nt, H, D)`` and ``(H, nt)``.
    """
    bias = _bias_table()
    qpos = seq_len - nt + torch.arange(nt)
    dist = qpos[:, None] - torch.arange(seq_len)[None, :]
    causal = dist >= 0
    has_bias = (dist >= 0) & (dist < REL)
    qb = q.permute(1, 0, 2).double()
    kb = k_seq.double().repeat_interleave(GRP, dim=0)
    vb = v_seq.double().repeat_interleave(GRP, dim=0)
    bmat = bias[:, dist.clamp(0, REL - 1).long()] * has_bias.unsqueeze(0)
    scores = qb @ kb.transpose(-1, -2) * SCALE + bmat
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    out = torch.softmax(scores, dim=-1) @ vb
    lse = torch.logsumexp(scores, dim=-1)
    return out.permute(1, 0, 2), lse


def _fake_rel_attention(k_seq, v_seq, seq_len, nt):
    """Stand-in for the Triton tail: biased attention over one key range."""

    def fake(q, k, v, out, lse, rel, **kwargs):
        k_lo = kwargs["k_lo_tok"]
        k_hi = min(kwargs["k_hi_tok"], seq_len)
        rows = q.shape[0]
        bias = _bias_table()
        qpos = seq_len - rows + torch.arange(rows)
        dist = qpos[:, None] - torch.arange(k_lo, k_hi)[None, :]
        causal = (dist >= 0) & (torch.arange(k_lo, k_hi) < seq_len)
        has_bias = causal & (dist < REL)
        qb = q.permute(1, 0, 2).double()
        kb = k_seq[:, k_lo:k_hi].double().repeat_interleave(GRP, dim=0)
        vb = v_seq[:, k_lo:k_hi].double().repeat_interleave(GRP, dim=0)
        bmat = bias[:, dist.clamp(0, REL - 1).long()] * has_bias.unsqueeze(0)
        scores = qb @ kb.transpose(-1, -2) * SCALE + bmat
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        out[:rows] = (torch.softmax(scores, dim=-1) @ vb).permute(1, 0, 2)
        lse[:, :rows] = torch.logsumexp(scores, dim=-1)

    return fake


def _run_prefill(layer, md, kv, nt, monkeypatch, sdpa_calls=None, rel_calls=None):
    """Drive `_prefill_attention` with the Triton tail faked and spies on."""
    seq_len = md.seq_lens[0].item()
    block_table = md.block_table[0]
    k_seq, v_seq = _seq_kv(kv, block_table, seq_len)

    if rel_calls is not None:
        monkeypatch.setattr(
            attention,
            "rel_attention",
            lambda *a, **k: (
                rel_calls.append((k.get("k_lo_tok"), k.get("k_hi_tok"))),
                _fake_rel_attention(k_seq, v_seq, seq_len, nt)(*a, **k),
            )[1],
        )
    if sdpa_calls is not None:
        real = attention._sdpa_flash

        def spy(q, k, v, scale):
            sdpa_calls.append(k.shape[2])
            return real(q, k, v, scale)

        monkeypatch.setattr(attention, "_sdpa_flash", spy)

    q = torch.randn(nt, H, D, dtype=torch.bfloat16)
    rel = torch.zeros((nt + 128, H, REL + 256), dtype=torch.bfloat16)
    out = torch.zeros(nt, H, D, dtype=torch.float32)
    lse = torch.zeros(H, nt, dtype=torch.float32)
    layer._prefill_attention(md, kv, q, rel, out, lse, 0)
    return q, k_seq, v_seq, out, lse


def test_merge_lse_matches_two_partial_reference():
    """`_merge_lse` merges (o, lse) partials like a single softmax pass."""
    torch.manual_seed(3)
    nt, n0, n1 = 8, 48, 32
    q = torch.randn(nt, H, D)
    k = torch.randn(n0 + n1, KVH, D)
    v = torch.randn(n0 + n1, KVH, D)
    kb = k.permute(1, 0, 2).repeat_interleave(GRP, dim=0)
    vb = v.permute(1, 0, 2).repeat_interleave(GRP, dim=0)
    scores = torch.einsum("thd,hkd->htk", q, kb) * SCALE
    s0, s1 = scores[..., :n0], scores[..., n0:]
    l0 = torch.logsumexp(s0, dim=-1)
    l1 = torch.logsumexp(s1, dim=-1)
    o0 = torch.einsum("htk,hkd->thd", torch.softmax(s0, dim=-1), vb[:, :n0])
    o1 = torch.einsum("htk,hkd->thd", torch.softmax(s1, dim=-1), vb[:, n0:])

    out = torch.zeros(nt, H, D)
    lse = torch.zeros(H, nt)
    out[:nt] = o0
    lse[:, :nt] = l0
    attention._merge_lse(out, lse, o1, l1, nt)

    ref = torch.logsumexp(scores, dim=-1)
    ref_o = torch.einsum("htk,hkd->thd", torch.softmax(scores, dim=-1), vb)
    assert torch.allclose(lse[:, :nt], ref, atol=1e-5)
    assert torch.allclose(out[:nt], ref_o, atol=1e-4, rtol=1e-4)


def test_sdpa_prefill_scratch_shared_bounded_and_page_aligned(monkeypatch):
    """One <=16 MiB K/V scratch per page shape, shared by every layer."""
    cpu = torch.device("cpu")
    buf = attention._sdpa_prefill_scratch(cpu, KVH, PAGE, D)
    assert buf is not None
    assert buf.dtype == torch.bfloat16
    assert buf.shape[0] == 2 and buf.shape[1] == KVH and buf.shape[3] == D
    # Window is page-aligned and within the <=16 MiB budget.
    assert buf.shape[2] % PAGE == 0
    assert buf.shape[2] <= 8192
    assert buf.nbytes <= attention._SDPA_PREFILL_SCRATCH_MAX_BYTES
    # K/V at full geometry fill the budget exactly: 4K-token windows.
    assert buf.shape == (2, KVH, 4096, D)
    assert buf.nbytes == 16 * 1024 * 1024
    # Same key -> same buffer; different page shape -> its own buffer.
    assert attention._sdpa_prefill_scratch(cpu, KVH, PAGE, D) is buf
    assert attention._sdpa_prefill_scratch(cpu, 4, PAGE, D) is not buf
    # Misaligned pages and meta devices reserve nothing.
    assert attention._sdpa_prefill_scratch(cpu, KVH, 12, D) is None
    assert attention._sdpa_prefill_scratch(torch.device("meta"), KVH, PAGE, D) is None

    monkeypatch.setattr(attention, "_SDPA_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)
    small = attention._sdpa_prefill_scratch(cpu, 2, PAGE, D)
    assert small.shape == (2, 2, 1024, D)


def test_hybrid_prefix_sdpa_matches_reference_attention(monkeypatch):
    """Chunked bias-free SDPA prefix + biased tail == full reference."""
    torch.manual_seed(11)
    seq_len, nt = 3000, 16
    npages = (seq_len + PAGE - 1) // PAGE + 2
    kv = _fp8_kv(npages)
    block_table = torch.randperm(npages, dtype=torch.int32)[
        : (seq_len + PAGE - 1) // PAGE
    ]
    layer = _make_layer()
    md = _make_md(seq_len, nt, block_table)

    # 4 MiB K/V scratch -> 1024-token (16-page) windows: the 30-page
    # prefix exercises the multi-window merge. Fresh dicts keep window
    # sizes independent of test order.
    monkeypatch.setattr(attention, "_SDPA_PREFILL_SCRATCH", {})
    monkeypatch.setattr(attention, "_SDPA_PREFILL_SCRATCH_MAX_BYTES", 4 << 20)
    monkeypatch.setattr(attention, "_FP8_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)
    monkeypatch.setattr(attention, "_HYBRID_PREFIX_SDPA", True)

    sdpa_calls, rel_calls = [], []
    q, k_seq, v_seq, out, lse = _run_prefill(
        layer, md, kv, nt, monkeypatch, sdpa_calls, rel_calls
    )

    k0_tok = seq_len - nt - REL + 1  # 1961
    sdpa_end = k0_tok // PAGE  # 30 pages
    end_page = (seq_len + PAGE - 1) // PAGE  # 47
    assert sdpa_calls == [16 * PAGE, 14 * PAGE]
    assert rel_calls and rel_calls[0][0] == sdpa_end * PAGE

    out_ref, lse_ref = _reference(q, k_seq, v_seq, seq_len, nt)
    rel_out = (out[:nt].double() - out_ref).norm() / out_ref.norm()
    rel_lse = (lse[:, :nt].double() - lse_ref).norm() / lse_ref.norm()
    assert rel_out < 3e-3
    assert rel_lse < 1e-4
    assert end_page > sdpa_end


def test_flag_off_keeps_byte_identical_stock_path(monkeypatch):
    """Default 0 never touches the SDPA arm or its scratch."""
    torch.manual_seed(5)
    seq_len, nt = 3000, 16
    npages = (seq_len + PAGE - 1) // PAGE + 2
    kv = _fp8_kv(npages)
    block_table = torch.randperm(npages, dtype=torch.int32)[
        : (seq_len + PAGE - 1) // PAGE
    ]
    layer = _make_layer()
    md = _make_md(seq_len, nt, block_table)

    monkeypatch.setattr(attention, "_FP8_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)
    monkeypatch.setattr(attention, "_HYBRID_PREFIX_SDPA", False)

    sdpa_calls, rel_calls = [], []
    _, _, _, out, lse = _run_prefill(
        layer, md, kv, nt, monkeypatch, sdpa_calls, rel_calls
    )
    end_page = (seq_len + PAGE - 1) // PAGE
    # Stock: the whole page range runs through rel_attention chunks.
    assert sdpa_calls == []
    assert rel_calls[0] == (0, min(4, end_page) * PAGE)


def test_hybrid_disabled_without_cpu_metadata_seq_len(monkeypatch):
    """The SDPA arm only engages on CPU-metadata seq len (zero .item())."""
    torch.manual_seed(7)
    seq_len, nt = 3000, 16
    npages = (seq_len + PAGE - 1) // PAGE + 2
    kv = _fp8_kv(npages)
    block_table = torch.randperm(npages, dtype=torch.int32)[
        : (seq_len + PAGE - 1) // PAGE
    ]
    layer = _make_layer()
    md = _make_md(seq_len, nt, block_table, cpu_len=False)

    monkeypatch.setattr(attention, "_FP8_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)
    monkeypatch.setattr(attention, "_HYBRID_PREFIX_SDPA", True)

    sdpa_calls, rel_calls = [], []
    _run_prefill(layer, md, kv, nt, monkeypatch, sdpa_calls, rel_calls)
    # seq len arrived via the device fallback -> stock chunked arm, no SDPA.
    assert sdpa_calls == []
    assert rel_calls and rel_calls[0] == (0, 4 * PAGE)


def test_hybrid_skipped_when_prefix_shorter_than_rel_extent(monkeypatch):
    """k0 <= 0 (no bias-free keys) leaves the stock path untouched."""
    torch.manual_seed(13)
    seq_len, nt = 400, 16  # 400 - 16 - 1024 + 1 < 0
    npages = (seq_len + PAGE - 1) // PAGE + 2
    kv = _fp8_kv(npages)
    block_table = torch.arange(npages, dtype=torch.int32)[
        : (seq_len + PAGE - 1) // PAGE
    ]
    layer = _make_layer()
    md = _make_md(seq_len, nt, block_table)

    monkeypatch.setattr(attention, "_FP8_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)
    monkeypatch.setattr(attention, "_HYBRID_PREFIX_SDPA", True)

    sdpa_calls, rel_calls = [], []
    _run_prefill(layer, md, kv, nt, monkeypatch, sdpa_calls, rel_calls)
    assert sdpa_calls == []
    end_page = (seq_len + PAGE - 1) // PAGE
    assert rel_calls == [(0, 4 * PAGE), (4 * PAGE, end_page * PAGE)]


def test_window_gather_and_merge_use_reserved_scratch(monkeypatch):
    """SDPA windows read slices of the one reserved buffer (no per-window
    allocations) and the gather lands pages in block-table order."""
    torch.manual_seed(17)
    seq_len, nt = 2000, 8
    npages = (seq_len + PAGE - 1) // PAGE + 2
    kv = _fp8_kv(npages)
    block_table = torch.randperm(npages, dtype=torch.int32)[
        : (seq_len + PAGE - 1) // PAGE
    ]
    layer = _make_layer()
    md = _make_md(seq_len, nt, block_table)

    # A 2 MiB scratch = 512-token (8-page) windows over the 15-page
    # prefix: 512-token then 448-token windows exercise the merge.
    monkeypatch.setattr(attention, "_SDPA_PREFILL_SCRATCH", {})
    monkeypatch.setattr(attention, "_SDPA_PREFILL_SCRATCH_MAX_BYTES", 2 << 20)
    monkeypatch.setattr(attention, "_FP8_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)
    monkeypatch.setattr(attention, "_HYBRID_PREFIX_SDPA", True)

    buf = attention._sdpa_prefill_scratch(torch.device("cpu"), KVH, PAGE, D)
    real = attention._sdpa_flash
    seen = []

    def spy(q, k, v, scale):
        seen.append(k.untyped_storage().data_ptr())
        return real(q, k, v, scale)

    monkeypatch.setattr(attention, "_sdpa_flash", spy)
    monkeypatch.setattr(
        attention,
        "rel_attention",
        lambda *a, **k: None,
    )

    q = torch.randn(nt, H, D, dtype=torch.bfloat16)
    rel = torch.zeros((nt + 128, H, REL + 256), dtype=torch.bfloat16)
    out = torch.zeros(nt, H, D, dtype=torch.float32)
    lse = torch.zeros(H, nt, dtype=torch.float32)
    layer._prefill_attention(md, kv, q, rel, out, lse, 0)

    k0_tok = seq_len - nt - REL + 1  # 969
    sdpa_end = k0_tok // PAGE  # 15 pages -> 512+448 tokens
    assert seen and len(set(seen)) == 1
    assert seen[0] == buf[0].untyped_storage().data_ptr()

    # Gather landed the right K values per logical page. Each window
    # reuses the scratch from offset 0, so the buffer's front holds the
    # second window (pages 8..14) and the tail of window 1 follows.
    def gather(page_idx):
        return (
            kv.view(torch.float8_e4m3fn)[block_table[page_idx].long()]
            .to(torch.bfloat16)[..., :D]
            .permute(1, 0, 2, 3)
            .reshape(KVH, -1, D)
        )

    assert torch.equal(buf[0][:, :448], gather(range(8, sdpa_end)))
    assert torch.equal(buf[0][:, 448:512], gather(range(0, 8))[:, 448:512])


def test_env_flag_defaults_to_off():
    """`INK2_HYBRID_PREFIX_SDPA` is opt-in; default keeps the stock path."""
    if os.environ.get("INK2_HYBRID_PREFIX_SDPA") == "1":
        pytest.skip("test env has the flag enabled")
    assert attention._HYBRID_PREFIX_SDPA is False

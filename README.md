# vllm-inkling2-xpu

Run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an
Intel Arc Pro B70 with [vLLM](https://github.com/vllm-project/vllm).

vLLM supports Inkling 2 Mini on NVIDIA GPUs only. This is an out-of-tree
[vLLM plugin](https://docs.vllm.ai/en/latest/design/plugin_system/) that adds
Intel XPU support: Triton kernels for the NVFP4/MXFP8 experts and attention, full
decode graphs, image inputs, and DSpark speculative decoding with the bundled
drafter. It registers itself through vLLM's `vllm.general_plugins` entry point
and does not patch vLLM.

Not affiliated with vLLM, Intel, or Thinking Machines.

## Tested configuration

| | |
|---|---|
| GPU | 1x Intel Arc Pro B70 (32 GB, Xe2 / Battlemage) |
| Host | Ubuntu 26.04.1, kernel 7.0.0-34, `xe` driver |
| vLLM | XPU build with Inkling 2 Mini ([#69028](https://github.com/vllm-project/vllm/pull/69028)) and the merged XPU FP32 LM-head path ([#69186](https://github.com/vllm-project/vllm/pull/69186)); the sparse-conv profile below pins an exact revision |
| PyTorch / Triton | 2.15.0+xpu / 3.9.0+xpu (pinned by vLLM's `requirements/xpu.txt`) |
| Model | `thinkingmachines/Inkling-2-Mini` (Apache-2.0, ~20 GB download) |

Other Battlemage cards are untested. The model needs ~18 GB of device memory for
weights, so a 24 GB card such as the B60 may only fit a short context.

## Quick start (Docker)

The steps below are the exact ones used to validate this README on a fresh host
with the Intel GPU driver installed (`ls /dev/dri/renderD*` lists your GPUs).

**1. Build vLLM's XPU image from `main`.** No published vLLM release or
`vllm/vllm-openai-xpu` nightly built before 2026-11-22 contains Inkling 2. Once a
nightly includes [#69028](https://github.com/vllm-project/vllm/pull/69028), you can
skip this step and use `BASE=vllm/vllm-openai-xpu:nightly` in step 2. A cold build
took about 17 minutes on a 72-core host.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# Use upstream main with the merged XPU LM-head fast path (#69186).
docker build -f docker/Dockerfile.xpu --target vllm-openai -t vllm-xpu:main .
cd ..
```

**2. Add the plugin.**

```bash
git clone https://github.com/mgaruccio/vllm-inkling2-xpu.git
docker build --build-arg BASE=vllm-xpu:main -t vllm-xpu:inkling2 vllm-inkling2-xpu
```

**3. Download the checkpoint** (~20 GB; optional, `vllm serve` downloads it
otherwise). Files the container writes into `~/.cache/huggingface` are owned by root.

```bash
mkdir -p ~/.cache/huggingface
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint hf vllm-xpu:inkling2 download thinkingmachines/Inkling-2-Mini
```

**4. Serve** on GPU 0 (`ZE_AFFINITY_MASK` selects the card).

```bash
docker run -d --name inkling2 \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host -p 8000:8000 -e ZE_AFFINITY_MASK=0 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-xpu:inkling2 \
  thinkingmachines/Inkling-2-Mini \
  --language-model-only --dtype bfloat16 \
  --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --gpu-memory-utilization 0.93 --block-size 64 --no-enable-prefix-caching \
  -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
docker logs -f inkling2   # ready when "Application startup complete" appears (~2 min)
```

Drop the `--speculative-config` line to run without speculative decoding.

**5. Send a request.** Inkling 2 reasons before it answers. The `inkling` reasoning
parser returns the reasoning in `message.reasoning` and the answer in
`message.content`. Leave room for both: if `max_tokens` runs out while the model is
still reasoning, `content` is `null` and `finish_reason` is `length`.
`reasoning_effort` shortens the reasoning.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "thinkingmachines/Inkling-2-Mini",
  "messages": [{"role": "user", "content": "Write a haiku about Intel GPUs."}],
  "max_tokens": 2048,
  "reasoning_effort": "low"
}' | python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

This returned a haiku in about 9 seconds (553 tokens, most of them reasoning).

### Try an image request

Rebuild the plugin image from current `main` if it predates vision support, then
recreate the server from step 4 **without `--language-model-only`**. The vision
encoder uses additional GPU memory; if the KV cache is too small at startup,
lower `--max-model-len`.

```bash
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
  "model": "thinkingmachines/Inkling-2-Mini",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/stop_sign.jpg"}},
    {"type": "text", "text": "What word is written on the sign?"}
  ]}],
  "max_tokens": 2048,
  "reasoning_effort": "low"
}' | python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

The expected answer is `STOP`.

## Install into an existing environment

Any XPU build of vLLM main at or after `c32251b5d1` works:

```bash
python -m pip install --no-deps git+https://github.com/mgaruccio/vllm-inkling2-xpu
```

Install it into the same Python environment that runs `vllm`. Check that it loaded:

```bash
python -c "from vllm.plugins import load_general_plugins as l; l(); \
from vllm.platforms.xpu import XPUPlatform as p; print('inkling2' in p.supported_quantization)"
```

## Serving flags

| Flag | Why |
|---|---|
| `--language-model-only` | Optional: disable the vision encoder for text-only serving and save GPU memory. Omit it to accept images. |
| `--dtype bfloat16` | BF16 activations (experts stay NVFP4/MXFP8). |
| `--max-model-len 8192` | The default context length does not fit on one B70. Raise it as far as your KV cache allows; the server log reports the KV cache size. |
| `-cc.mode=none` | `torch.compile` is not supported for this model on XPU. |
| `-cc.cudagraph_mode=FULL_DECODE_ONLY` | Decode graphs (the name says CUDA; it applies to XPU). |
| `--max-num-seqs 1` | Conservative quick-start default; C32 was verified separately with the sparse-conv stack below. |
| `--reasoning-parser inkling`, `--tool-call-parser inkling` | Same parsers as on NVIDIA. |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":7}'` | DSpark with the bundled `mtp.safetensors` drafter. DSpark requires exactly 7. |

### Optional: 409K text-only serving with DSpark and sparse conv history

Use the [official XPU source-install instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#intel-xpu) with vLLM pinned as follows. Both commits of [#69444](https://github.com/vllm-project/vllm/pull/69444) are needed; #69186 is already in the base.

```bash
git clone https://github.com/vllm-project/vllm.git
git -C vllm checkout --detach 0085facc58dedeef757aca3f51d6508e2ef9b148
git -C vllm fetch origin refs/pull/69444/head
git -C vllm cherry-pick 0811005c59b750e63f9bd6658842fa7d529b8e21 34a0975f5c9dcee1c88742aacb38de841c3eb7e2
```

Install the plugin at `677db18d0129703b529270fe75052272f0e2834a` or newer into the same XPU environment. [Build the CUTLASS Xe2 extension](#optional-cutlass-xe2-expert-kernel-ink2_cutlass_moe1) before enabling it. The verified replay used PyTorch 2.15.0+xpu, Triton 3.9.0+xpu, XPU kernels 0.1.17, and compute-runtime 26.40 on one 32 GB B70, with [Inkling-2-Mini `ed40e7da`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7).

```bash
INK2_SPARSE_CONV_HISTORY=1 INK2_CUTLASS_MOE=1 INK2_HYBRID_PREFIX_SDPA=1 \
  vllm serve /path/to/Inkling-2-Mini --host 127.0.0.1 \
  --language-model-only --dtype bfloat16 --kv-cache-dtype fp8 \
  --max-model-len 409000 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --block-size 64 --gpu-memory-utilization 0.98 \
  --enable-prefix-caching --no-async-scheduling \
  -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

The flag is off by default and requires the patched core. It changes how the short convolutions retain history, not the attention KV format. Keep default prefix retention and no KV connector or microbatching for this recipe. Lower `--max-model-len` if the engine cannot reserve enough KV on your host; do not disable the memory check.

In [development validation](https://github.com/mgaruccio/b70-inference/blob/16e94ad4/results/20261125-inkling2-sparse-conv/README.md), the sparse path improved the effective cache estimate by 22% at the same C32/320K budget. A separate C32/409K deployment completed a **408,744-token input plus 128 output tokens** with zero preemptions and DSpark active. This establishes capacity for one near-limit request, not 32 such requests or long-context answer quality. The pool is shared; don't expect interactive latency alongside a long prefill.

### Optional: FP8 KV and hybrid XPU flash prefill

Use `--kv-cache-dtype fp8` (E4M3) to reduce KV memory. Set
`INK2_HYBRID_PREFIX_SDPA=1` to also route the page-aligned, bias-free prefix of
eligible single-request global-attention prefill through XPU flash SDPA
(`aten._scaled_dot_product_flash_attention`, the SYCL-TLA backend, not oneDNN).
The relative-bias tail still runs on Triton; the partial results are merged using
FP32 log-sum-exp. This applies to prefill, not decode.

The path reuses a BF16 K/V scratch buffer (up to 16 MiB per device and cache
geometry); gather, kernel, and merge temporaries can still allocate. Disabling
the flag (the default), missing CPU sequence-length metadata, or an ineligible
prefix uses the original attention path. The flag requires working XPU flash
attention support; kernel failures are not caught and retried on Triton. To
roll back the optimization, restart with `INK2_HYBRID_PREFIX_SDPA=0`.

On a single B70, a matched development run with FP8 KV, CUTLASS and DSpark k=7
measured the following median cold-prefill rates (three requests per depth):

| Prompt tokens | Hybrid off (tok/s) | Hybrid on (tok/s) | Change |
|---:|---:|---:|---:|
| 65,536 | 3,108.0 | 3,436.9 | +10.6% |
| 131,072 | 2,461.8 | 2,851.7 | +15.8% |
| 245,760 | 1,812.3 | 2,203.5 | +21.6% |

[Benchmark method, raw measurements and live configuration](https://github.com/mgaruccio/b70-inference/tree/main/results/20261124-inkling2-hybrid-prefix-sdpa).

## Performance

Development-tier measurements on one Arc Pro B70, single stream, greedy, 256
output tokens, 6 coding and reasoning chat prompts. These are not standardized
benchmark results; raw results and the script are in [`results/`](results/).

Measured 2026-11-23 on vLLM main `3146a9df41` + #69186, with `--max-model-len 32768 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92` and a 275 W cap. Throughput figures are decode tok/s (median); acceptance length is shown in parentheses where DSpark is on.

| | Default kernels | `INK2_CUTLASS_MOE=1` (see below) |
|---|---:|---:|
| No speculation | **57.6** | 56.0 |
| No speculation, +6K / +30K tokens of context | **54.9 / 51.3** | 53.4 / 50.0 |
| DSpark k=7 | 83.0 (2.99) | **99.0 (3.16)** |
| DSpark k=7, +6K / +30K context | 82.2 (3.22) / 76.4 (3.11) | **103.7 (3.13) / 100.7 (3.57)** |
| 8K cold prefill, tok/s | 3,615 | **4,085** |
| Top-1 vs CPU fp32 (4,174 positions; CPU bf16 floor 0.9938) | 0.9938 | 0.9938 |

Raw summaries and launch lines are in [`results/decode/cutlass-xe2-20261123/`](results/decode/cutlass-xe2-20261123/). Without #69186, decode is about 7.5 ms/token slower.

DSpark gains depend on the workload. On short chat prompts with reasoning disabled, acceptance length was 2.02 and DSpark was slower than no speculation. The first requests after startup are slower while Triton
kernels compile.

With DSpark enabled, greedy output can differ from greedy output without it.
Verification runs several tokens through the batched expert kernel instead of the
single-token kernel, so rounding differs and near-tied tokens can flip; in the
cases we checked, the top-two log-probabilities at the first divergence were 0.025
to 0.068 apart.

### Optional: CUTLASS Xe2 expert kernel (`INK2_CUTLASS_MOE=1`)

The routed experts can run on a SYCL CUTLASS grouped GEMM instead of the Triton kernels. It reads the checkpoint's NVFP4 bytes directly for layers 0–19 and uses the vllm-xpu-kernels MXFP8 grouped GEMM for layers 20–23.

**When to use it:**

- It gives about 20–30% faster DSpark decode and 13% faster prefill.
- Non-speculative decode is about 3% slower, so leave it off for servers that don't use DSpark.
- Quality: no measured loss (table above).

**Credit and license:** the kernel and its dispatcher are by [SergiioB](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook), vendored under the MIT license (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

**Build.** It is a compiled extension, built once into the installed plugin. You need Intel oneAPI `icpx` (tested 2026.1.1), `ocloc` from the [compute-runtime release](https://github.com/intel/compute-runtime/releases) matching your driver, and header checkouts of [sycl-tla](https://github.com/intel/sycl-tla) `7b56b936` and [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) `e9b6e4b2`:

```bash
source /opt/intel/oneapi/setvars.sh
PYTHON=/path/to/venv/bin/python SYCL_TLA_DIR=/path/to/sycl-tla \
  VLLM_XPU_KERNELS_DIR=/path/to/vllm-xpu-kernels scripts/build-cutlass-xe2.sh
```

**Serve.** Start `vllm serve` with `INK2_CUTLASS_MOE=1` in the environment, from a shell where oneAPI's `setvars.sh` has **not** been sourced. Its runtime libraries would shadow the ones bundled with PyTorch.

**Not compatible with:** kernel-format weight reloads.

## Numerical parity

Compared against an independent reference: Hugging Face Transformers Inkling 2 on
CPU, experts dequantized from the same checkpoint to fp32, BF16 activations
(matching the XPU semantics). Teacher-forced next-token log-probabilities on 4
short prompts and the first 2,048 positions of 2 long prompts (4,174 positions):

| vs CPU fp32 | top-1 agreement | median / p95 / max abs Δ logprob |
|---|---|---|
| XPU | 99.4% | 0.00017 / 0.024 / 5.69 |
| CPU bf16 (noise floor) | 99.4% | 0.00017 / 0.029 / 5.63 |

Short prompts individually agree 96.2–100% on top-1 tokens. A 64-token greedy chat
continuation matched the CPU fp32 reference exactly. The largest single layer
difference found was a router near-tie (8th vs 9th expert scores 6e-5 apart).
The XPU side was an earlier in-tree build of the same kernels (vLLM branch commit
`32ecd66097`); later builds and this plugin produced token-identical greedy output
on the validation prompts. This is bounded evidence on sampled sequences, not a
bitwise or general guarantee. Details: [`results/parity.md`](results/parity.md).

## Limitations

- One device: tensor and expert parallelism raise an error at startup.
- BF16 activations, no LoRA, no `torch.compile`.
- Expert activations stay BF16 (W4A16/W8A16), like vLLM's Hopper path; the
  checkpoint's NVFP4/MXFP8 activation quantization is not applied.
- Validated with one sequence at a time.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `inkling2 quantization is currently not supported in xpu` | The plugin is not installed in the Python environment that runs `vllm`. |
| `Model architectures ['Inkling2LoopForConditionalGeneration'] are not supported` | vLLM is older than `c32251b5d1`; build from current `main`. |
| `Free memory on device ... is less than desired GPU memory utilization` | Another process is using the GPU, or lower `--gpu-memory-utilization`. |
| Out of memory or KV cache too small | Lower `--max-model-len`. |
| Using another GPU | Set `ZE_AFFINITY_MASK=<index>`; one card per server. |

## Tests

From a vLLM source checkout, after installing the plugin, on an XPU machine:

```bash
PYTHONPATH="$PWD:/path/to/vllm-inkling2-xpu/tests" \
  pytest -q -p tests.conftest /path/to/vllm-inkling2-xpu/tests
```

`test_xpu_cutlass_experts` runs only after `scripts/build-cutlass-xe2.sh`, and otherwise skips.

## Compatibility

The plugin uses vLLM's documented `vllm.general_plugins` entry point and
`ModelRegistry.register_model` API, and appends `inkling2` to
`vllm.platforms.xpu.XPUPlatform.supported_quantization`. It imports the shared
`vllm.models.inkling2.{configs,processing,quant_config,speculator_config}`
modules, `vllm.models.inkling.nvidia.sconv_swa_attn`, and vLLM internals in
`vllm.config`, `vllm.forward_context`, `vllm.model_executor`, `vllm.multimodal`,
`vllm.triton_utils`, `vllm.v1`, `vllm.distributed`, `vllm.envs`, `vllm.sequence`,
and `vllm.utils`. Changes to these may require a plugin update.

## License

Apache-2.0. Most of the code is derived from vLLM's Inkling 2 implementation and
keeps its copyright headers.

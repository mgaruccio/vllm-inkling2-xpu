# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Inkling 2 Mini NVFP4 CUTLASS experts on Intel XPU.

Use the same script and pinned XPU environment for each A/B/A run, changing only
which plugin is on PYTHONPATH (including its built CUTLASS Xe2 extension):

    PYTHONPATH=/path/to/variant/src python scripts/bench_verify_nvfp4.py -o A1.json

Compare output_sha256 and median_us by fixture across fresh processes. Timings
are synchronized graph replays, not end-to-end verifier timings. Logical bytes
count one read of each live expert's weights/scales plus fixture inputs and
output, excluding intermediate buffers and any rereads; they are not DRAM bytes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter_ns

# Inkling 2 Mini checkpoint dimensions and activation parameters (test_moe_xpu.py).
EXPERTS, HIDDEN, INTERMEDIATE, TOPK = 128, 2304, 1152, 8
GATE_CAP, UP_CAP = 4.0, 25.0
SEED = 291
SHAPES = (("c1", 8), ("c8", 32), ("r128", 128), ("r1", 1))


def _projection(n, k, rng, device):
    import torch

    # Checkpoint layout, without the Triton split-half repack. All E2M1 nibble
    # patterns are finite; 0x28..0x44 are positive, finite E4M3FN scale codes.
    packed = torch.randint(
        256, (EXPERTS, n, k // 2), generator=rng, dtype=torch.uint8
    ).to(device)
    scale_codes = torch.randint(
        0x28, 0x45, (EXPERTS, n, k // 16), generator=rng, dtype=torch.uint8
    )
    amax = 0.5 + torch.rand(EXPERTS, generator=rng, dtype=torch.float32)
    outer = (amax / (2 * math.sqrt(k))).to(device)
    return packed, scale_codes.to(device).view(torch.float8_e4m3fn), outer


def _inputs(rows, routing, device):
    import torch

    # The same activations and mixing weights are used for both routing patterns.
    rng = torch.Generator(device="cpu").manual_seed(SEED + rows)
    x = torch.randn(rows, HIDDEN, generator=rng, dtype=torch.float32).bfloat16()
    padding = torch.arange(rows) % 7 == 3
    experts = torch.randperm(EXPERTS, generator=rng)
    # Concentrated: one hot set of eight for all live rows. Spread: cycle through
    # disjoint sets of eight, covering all 128 experts when 16 live rows fit.
    if routing == "spread":
        groups = (~padding).cumsum(0)
    else:
        groups = torch.zeros_like(padding, dtype=torch.long)
    groups[padding] = EXPERTS // TOPK - 1
    ids = experts[(groups[:, None] * TOPK + torch.arange(TOPK)) % EXPERTS]
    scores, order = torch.randn(rows, TOPK, generator=rng, dtype=torch.float32).sort(
        dim=-1, descending=True
    )
    ids = ids.gather(1, order).int().contiguous()
    scores = scores.sigmoid()
    weights = (scores / scores.sum(-1, keepdim=True) * TOPK).contiguous()
    # Padded rows still carry finite activations, valid IDs and nonzero weights.
    unique_experts = ids[~padding].unique().numel()
    # The one-row fixture is a live token with no mask, as in the kernel test.
    mask = padding.to(device) if rows > 1 else None
    inputs = x.to(device), ids.to(device), weights.to(device), mask
    return inputs, padding, unique_experts


def _digest(out, x, padding, name):
    import torch

    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise RuntimeError(f"{name}: expected BF16 XPU output matching the input shape")
    host = out.cpu().contiguous()
    if not host.isfinite().all().item():
        raise RuntimeError(f"{name}: non-finite expert output")
    if (host[padding] != 0).any().item():
        raise RuntimeError(f"{name}: padded output rows must be zero")
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()


def _bench_fixture(kernel, projections, stream, shape, rows, routing, args):
    import torch

    name = f"{shape}_{routing}"
    inputs, padding, unique_experts = _inputs(rows, routing, stream.device)

    def forward():
        return kernel(*inputs, *projections, True, EXPERTS, GATE_CAP, UP_CAP)

    # Warm the eager path (including compilation) before capturing one forward.
    eager = forward()
    for _ in range(args.warmup - 1):
        eager = forward()
    expected = _digest(eager, inputs[0], padding, name)

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        out = forward()
    for _ in range(args.warmup):
        graph.replay()
    if _digest(out, inputs[0], padding, name) != expected:
        raise RuntimeError(f"{name}: XPU graph and eager output bits differ")

    samples_us = []
    stream.synchronize()
    for _ in range(args.repeats):
        start = perf_counter_ns()
        for _ in range(args.replays):
            graph.replay()
        stream.synchronize()
        samples_us.append((perf_counter_ns() - start) / (1000 * args.replays))

    if _digest(out, inputs[0], padding, name) != expected:
        raise RuntimeError(f"{name}: output bits changed during graph replays")
    bytes_per_expert = sum(t.nbytes for t in projections) // EXPERTS
    io_bytes = sum(t.nbytes for t in (*inputs, out) if t is not None)
    return {
        "fixture": name,
        "rows": rows,
        "routing": routing,
        "padded_rows": padding.sum().item(),
        "unique_experts": unique_experts,
        "logical_bytes": bytes_per_expert * unique_experts + io_bytes,
        "output_sha256": expected,
        "median_us": median(samples_us),
        "samples_us": samples_us,
    }


def _run(args):
    import torch

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise SystemExit("This benchmark requires an Intel XPU with PyTorch XPU support.")
    # Only the variant selected by normal PYTHONPATH imports is loaded.
    from vllm_inkling2_xpu.ops import moe_cutlass

    device = torch.device("xpu", torch.xpu.current_device())
    stream = torch.xpu.Stream(device=device)
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.inference_mode(), torch.xpu.stream(stream):
        # Reuse the full set of weights for every fixture (about 547 MiB).
        projections = (
            *_projection(2 * INTERMEDIATE, HIDDEN, rng, device),
            *_projection(HIDDEN, INTERMEDIATE, rng, device),
        )
        kernel = moe_cutlass.cutlass_experts
        fixtures = [
            _bench_fixture(kernel, projections, stream, shape, rows, routing, args)
            for shape, rows in SHAPES
            for routing in ("concentrated", "spread")
        ]
    return {
        "implementation": str(Path(moe_cutlass.__file__).resolve()),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "device_name": torch.xpu.get_device_name(device),
        "seed": SEED,
        "num_experts": EXPERTS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "topk": TOPK,
        "timing": {
            "method": "synchronized_xpu_graph_replay_wall",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "replays_per_repeat": args.replays,
        },
        "fixtures": fixtures,
    }


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--warmup", type=_positive_int, default=3, help="warmups per mode")
    parser.add_argument("--repeats", type=_positive_int, default=5, help="timed batches")
    parser.add_argument("--replays", type=_positive_int, default=16, help="replays per batch")
    args = parser.parse_args()
    text = json.dumps(_run(args), indent=2, allow_nan=False)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

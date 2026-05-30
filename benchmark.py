from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from functools import partial


def make_airport_workload(jax, jnp):
    @partial(jax.jit, static_argnames=("steps",))
    def airport_workload(left, right, *, steps: int):
        scale = jax.lax.rsqrt(jnp.asarray(left.shape[0], dtype=left.dtype))

        value = left
        for _ in range(steps):
            value = (value @ right) * scale
            value = jnp.tanh(value)

        return value

    return airport_workload


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the extremely official JAX laptop duel benchmark."
    )
    parser.add_argument("--size", type=positive_int, default=1536)
    parser.add_argument("--steps", type=positive_int, default=8)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "mps"),
        default="cpu",
        help="JAX backend to initialize. Use mps with the jax-mps extra on Apple Silicon.",
    )
    return parser.parse_args()


def configure_backend(backend: str) -> None:
    if backend == "mps":
        os.environ["JAX_PLATFORMS"] = "mps,cpu"
    elif backend != "auto":
        os.environ["JAX_PLATFORMS"] = backend


def import_jax(backend: str):
    try:
        import jax
        import jax.numpy as jnp
    except Exception as error:
        if backend == "mps":
            print(
                "Could not import JAX with the MPS backend. Try:\n"
                "  uv run --locked --python 3.13 --extra mps python benchmark.py --backend mps",
                file=sys.stderr,
            )
            raise SystemExit(2) from error
        raise

    try:
        jax.devices()
    except Exception as error:
        if backend == "mps":
            print(
                "Could not initialize JAX MPS. Make sure this is macOS on Apple Silicon "
                "with Python 3.13 and the mps extra installed:\n"
                "  uv run --locked --python 3.13 --extra mps python benchmark.py --backend mps",
                file=sys.stderr,
            )
            raise SystemExit(2) from error
        raise

    return jax, jnp


def get_target_device(jax, backend: str):
    if backend == "auto":
        return jax.devices()[0]
    return jax.devices(backend)[0]


def make_random_inputs(jax, jnp, args: argparse.Namespace, target_device):
    key = jax.random.PRNGKey(args.seed)
    left_key, right_key = jax.random.split(key)
    shape = (args.size, args.size)

    if args.backend == "mps":
        random_device = jax.devices("cpu")[0]
    else:
        random_device = target_device

    with jax.default_device(random_device):
        left = jax.random.normal(left_key, shape, dtype=jnp.float32)
        right = jax.random.normal(right_key, shape, dtype=jnp.float32)

    if random_device != target_device:
        left = jax.device_put(left, target_device)
        right = jax.device_put(right, target_device)

    return left, right


def main() -> None:
    args = parse_args()
    configure_backend(args.backend)
    jax, jnp = import_jax(args.backend)
    airport_workload = make_airport_workload(jax, jnp)
    device = get_target_device(jax, args.backend)

    left, right = make_random_inputs(jax, jnp, args, device)
    jax.block_until_ready((left, right))

    warmup_start = time.perf_counter()
    result = airport_workload(left, right, steps=args.steps)
    jax.block_until_ready(result)
    warmup_seconds = time.perf_counter() - warmup_start

    times: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = airport_workload(result, right, steps=args.steps)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)

    matmul_flops = 2 * args.size**3
    total_flops = matmul_flops * args.steps
    best_gflops = total_flops / min(times) / 1_000_000_000
    avg_gflops = total_flops / statistics.fmean(times) / 1_000_000_000
    checksum = float(jnp.mean(jnp.abs(result)))

    device_kind = getattr(device, "device_kind", type(device).__name__)
    run_times = ", ".join(f"{seconds:.3f}s" for seconds in times)

    print("JAX Airport Laptop Duel")
    print("=======================")
    print(f"Python: {platform.python_version()}")
    print(f"JAX: {jax.__version__}")
    print(f"Backend: {jax.default_backend()} ({device_kind})")
    print(f"Matrix: {args.size} x {args.size}, float32")
    print(f"Steps per repeat: {args.steps}")
    print(f"Repeats: {args.repeats}")
    print(f"Warmup/compile: {warmup_seconds:.3f}s")
    print(f"Run times: {run_times}")
    print(f"Checksum: {checksum:.6f}")
    print()
    print(f"BENCHMARK SCORE: {best_gflops:.2f} GFLOP/s")
    print(f"Average score: {avg_gflops:.2f} GFLOP/s")


if __name__ == "__main__":
    main()

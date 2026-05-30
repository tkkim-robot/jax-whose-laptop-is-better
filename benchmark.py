from __future__ import annotations

import argparse
import platform
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("steps",))
def airport_workload(left: jax.Array, right: jax.Array, *, steps: int) -> jax.Array:
    scale = jax.lax.rsqrt(jnp.asarray(left.shape[0], dtype=left.dtype))

    def body(_: int, value: jax.Array) -> jax.Array:
        value = (value @ right) * scale
        return jnp.tanh(value)

    return jax.lax.fori_loop(0, steps, body, left)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    key = jax.random.PRNGKey(args.seed)
    left_key, right_key = jax.random.split(key)
    shape = (args.size, args.size)

    left = jax.random.normal(left_key, shape, dtype=jnp.float32)
    right = jax.random.normal(right_key, shape, dtype=jnp.float32)
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

    device = jax.devices()[0]
    run_times = ", ".join(f"{seconds:.3f}s" for seconds in times)

    print("JAX Airport Laptop Duel")
    print("=======================")
    print(f"Python: {platform.python_version()}")
    print(f"JAX: {jax.__version__}")
    print(f"Backend: {jax.default_backend()} ({device.device_kind})")
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

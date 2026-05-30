# JAX Whose Laptop Is Better?

TK and Daniel were sitting in the New Orleans airport after the ACC 2026 trip, waiting to board, staring at each other's laptops with the calm seriousness of people who had absolutely discovered the most important scientific question:

**whose Apple Silicon laptop can make JAX do math faster?**

This repo is the tiny, questionably official answer machine. Clone it, run one command, compare the score, and then behave with exactly the amount of dignity your number deserves.

## Setup

Install `uv` on macOS or Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Official instructions live at [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/), in case your terminal wants to be dramatic.

The default CPU duel uses only `jax`. The optional Apple Silicon GPU lane adds `jax-mps`, because tiny suitcase, slightly suspicious engine upgrade. The `uv.lock` file is committed so everyone boards with the same dependency luggage.

## Run The Duel

```sh
uv run --locked python benchmark.py
```

The script prints your JAX backend, run times, checksum, and one glorious benchmark score in GFLOP/s. Bigger number wins. Smaller number buys airport snacks.

## Apple Silicon GPU Duel

For the experimental MPS path, use `jax-mps`. It currently wants macOS on Apple Silicon, Python 3.13, and the `mps` JAX platform:

```sh
uv run --locked --python 3.13 --extra mps python benchmark.py --backend mps
```

To force the CPU lane for comparison:

```sh
uv run --locked python benchmark.py --backend cpu
```

Optional knobs, for people who believe a boarding gate can also be a lab:

```sh
uv run --locked python benchmark.py --size 2048 --steps 10 --repeats 7
```

Use the same settings as your friends or the victory speech gets thrown out by the imaginary benchmark committee.

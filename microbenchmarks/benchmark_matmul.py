r"""Benchmark for matrix multiplication.

Sample usage (on TPU vm):
  $ python benchmark_matmul.py \
  --dim 4096 4096 4096 \
  --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 \
  --matcher="jit_matmul.*"
"""

import argparse
import os
import re
from typing import Any
from benchmark_utils import run_bench
import jax
import jax.numpy as jnp


def matmul(a, b):
  return a @ b


def get_dtype(dtype: str):
  if dtype == "bf16":
    return jnp.bfloat16
  if dtype == "fp8_e5m2":
    return jnp.float8_e5m2
  if dtype == "fp8_e4m3":
    return jnp.float8_e4m3fn
  raise ValueError(f"Invalid data type: {dtype}")


def main():
  """Benchmark for matrix multiplication."""
  parser = argparse.ArgumentParser(
      description="Run matrix multiplication benchmark."
  )

  parser.add_argument(
      "--dtype",
      type=str,
      choices=["bf16", "fp8_e5m2", "fp8_e4m3"],
      default="bf16",
      help="Data type of the matrix elements.",
  )
  parser.add_argument(
      "--libtpu_args",
      type=str,
      required=False,
      help=(
          "LIBTPU_INIT_ARGS environment variable, e.g."
          " '--xla_tpu_scoped_vmem_limit_kib=65536'."
      ),
  )
  parser.add_argument(
      "--dim",
      type=int,
      required=True,
      nargs=3,
      help=(
          "Dimensions of the two matices to be multiplied. 3 integers, m, n, k"
          " must be specified, implying (m, n) * (n, k)."
      ),
  )
  parser.add_argument(
      "--num_iter",
      type=int,
      default=100,
      help="Number of times the matmul kernel will be run.",
  )
  parser.add_argument(
      "--warmup_iter",
      type=int,
      default="1",
      help=(
          "Number of times the matmul kernel will be run to warm up before the"
          " actual timing measurement starts."
      ),
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/matmul",
      help="The directory to save the profiler trace to.",
  )
  parser.add_argument(
      "--label",
      type=str,
      default="my_func",
      help=(
          "A label used to name the function to be benchmarked in the trace"
          " events."
      ),
  )
  parser.add_argument(
      "--matcher",
      type=str,
      required=False,
      help=(
          "A regex-based string matcher to filter the trace events eligible for"
          " benchmarking. This arg would be useful if we want to measure the"
          " timing of a specific op or XLA module within the function., e.g."
          " --matcher='fusion' measures the timing of XLA fusion op"
          " specifically."
      ),
  )

  args = parser.parse_args()

  if args.libtpu_args:
    os.environ["LIBTPU_INIT_ARGS"] = args.libtpu_args

  dtype = get_dtype(args.dtype)
  m, n, k = args.dim[0], args.dim[1], args.dim[2]
  a = jax.random.normal(jax.random.key(0), (m, n)).astype(dtype)
  b = jax.random.normal(jax.random.key(0), (n, k)).astype(dtype)

  compiled = jax.jit(matmul).lower(a, b).compile()
  matcher = re.compile(args.matcher) if args.matcher else None
  result = run_bench(
      lambda: jax.block_until_ready(compiled(a, b)),
      num_iter=args.num_iter,
      warmup_iter=args.warmup_iter,
      log_dir=args.log_dir,
      func_label=args.label,
      event_matcher=matcher,
  )

  # 2 ops (multiply and add)
  compute = m * n * k * 2
  tflops = compute / result.time_median / 1e12

  print(
      f"dtype: {dtype.__name__}, matrix dimensions: ({m}, {n}, {k}), time taken"
      f" (ms, median): {result.time_median * 1e3}, TFLOPS: {tflops}"
  )


if __name__ == "__main__":
  main()
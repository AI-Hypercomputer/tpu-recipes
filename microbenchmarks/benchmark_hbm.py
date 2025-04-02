r"""Benchmark for HBM bandwidth.

Sample usage (on TPU vm):
  $ python benchmark_hbm.py \
  --num_elements=16777216 \
  --matcher="jit_my_copy.*"
"""

import argparse
import os
import re
from typing import Any
from benchmark_utils import run_bench
import jax
import jax.numpy as jnp


def my_copy(a):
  return a.copy()


def get_dtype(dtype: str):
  if dtype == "bf16":
    return jnp.bfloat16
  if dtype == "fp8_e5m2":
    return jnp.float8_e5m2
  if dtype == "fp8_e4m3":
    return jnp.float8_e4m3fn
  raise ValueError(f"Invalid data type: {dtype}")


def main():
  """Benchmark for HBM bandwidth."""

  parser = argparse.ArgumentParser(
      description="Run HBM bandwidth benchmark."
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
      "--num_elements",
      type=int,
      required=True,
      help="Number of elements in the array.",
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
          " acutal timing measurement starts."
      ),
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/hbm",
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
  n = args.num_elements
  a = jax.random.normal(jax.random.key(0), (n,)).astype(dtype)
  compiled = jax.jit(my_copy).lower(a).compile()

  matcher = re.compile(args.matcher) if args.matcher else None
  result = run_bench(
      lambda: jax.block_until_ready(compiled(a)),
      num_iter=args.num_iter,
      warmup_iter=args.warmup_iter,
      log_dir=args.log_dir,
      func_label=args.label,
      event_matcher=matcher,
  )

  tensor_size = n * a.itemsize
  bw_gbps = (tensor_size * 2) / result.time_median / 1e9  # read + write = 2

  print(
      f"Tensor size (bytes): {tensor_size}, time taken (ms, median):"
      f" {result.time_median * 1000}, bandwidth (GBps, median): {bw_gbps} "
  )


if __name__ == "__main__":
  main()

r"""Benchmark for HBM bandwidth.

Sample usage (on TPU vm):
  $ python benchmark_hbm.py
"""

import argparse
import os
import re
from typing import Any
from benchmark_utils import run_bench
import jax
import jax.numpy as jnp
import pandas as pd


D_TYPE = jnp.bfloat16

NUM_ELEMENTS = [2**i for i in range(10, 33)]

NUM_ITER = 100
WARMUP_ITER = 1
FUNC_LABEL = "my_func"
EVENT_MATCHER = "jit_my_copy.*"  # If None, function runtime is measured
# EVENT_MATCHER = None
LOG_DIR = "/tmp/hbm"
OUTPUT = "/tmp/hbm_result.tsv"


def my_copy(a):
  return a.copy()


def main():
  """Benchmark for HBM bandwidth."""
  results = []
  for n in NUM_ELEMENTS:
    a = jax.random.normal(jax.random.key(0), (n,)).astype(D_TYPE)
    compiled = jax.jit(my_copy).lower(a).compile()
    matcher = re.compile(EVENT_MATCHER) if EVENT_MATCHER else None
    result = run_bench(
        lambda: jax.block_until_ready(compiled(a)),
        num_iter=NUM_ITER,
        warmup_iter=WARMUP_ITER,
        log_dir=LOG_DIR,
        func_label=FUNC_LABEL,
        event_matcher=matcher,
    )

    tensor_size = n * a.itemsize
    bw_gbps = (tensor_size * 2) / result.time_median / 1e9  # read + write = 2
    results.append({
        "dtype": D_TYPE.__name__,
        "tensor_size_bytes": tensor_size,
        "time_median": result.time_median,
        "bandwidth_gbps_median": bw_gbps,
        "time_min": result.time_min,
        "bandwidth_gbps_max": (tensor_size * 2) / result.time_min / 1e9,
    })

    print(
        f"Tensor size: {tensor_size / 1024**2} MB, time taken (median):"
        f" {result.time_median * 1000:.4f} ms, bandwidth: {bw_gbps:.2f} GBps"
    )

  pd.DataFrame(results).to_csv(OUTPUT, sep="\t", index=False)


if __name__ == "__main__":
  main()

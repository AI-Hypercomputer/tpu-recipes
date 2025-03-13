r"""Benchmark for matrix multiplication.

Sample usage (on TPU vm):
  $ python benchmark_matmul.py
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

MAT_DIMS = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (32768, 32768, 32768),
    (65536, 65536, 65536),
    (16384, 8192, 1280),
    (16384, 1024, 8192),
    (16384, 8192, 7168),
    (16384, 3584, 8192),
    (8192, 8192, 8192),
    (1536, 1024, 24576),
    (1024, 24576, 1536),
    (16384, 24576, 5120),
]

NUM_ITER = 100
WARMUP_ITER = 1
FUNC_LABEL = "my_func"
EVENT_MATCHER = "jit_matmul.*"  # If None, function runtime is measured
# EVENT_MATCHER = None
LOG_DIR = "/tmp/matmul"
OUTPUT = "/tmp/matmul_result_fp8.tsv"


def matmul(a, b):
  return a @ b


def main():
  """Benchmark for matrix multiplication."""
  os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=65536"

  results = []
  for m, n, k in MAT_DIMS:
    a = jax.random.normal(jax.random.key(0), (m, n)).astype(D_TYPE)
    b = jax.random.normal(jax.random.key(0), (n, k)).astype(D_TYPE)
    compiled = jax.jit(matmul).lower(a, b).compile()
    matcher = re.compile(EVENT_MATCHER) if EVENT_MATCHER else None
    result = run_bench(
        lambda: jax.block_until_ready(compiled(a, b)),
        num_iter=NUM_ITER,
        warmup_iter=WARMUP_ITER,
        log_dir=LOG_DIR,
        func_label=FUNC_LABEL,
        event_matcher=matcher,
    )

    # 2 ops (multiply and add)
    ops = m * n * k * 2
    results.append({
        "dtype": D_TYPE.__name__,
        "dimensions": (m, n, k),
        "time_median": result.time_median,
        "tflops_median": ops / result.time_median / 1e12,
        "time_min": result.time_min,
        "tflops_max": ops / result.time_min / 1e12,
        "time_fn_median": result.time_fn_median,
        "tflops_fn_median": ops / result.time_fn_median / 1e12,
    })

    print(
        f"dtype: {D_TYPE.__name__}, matrix Dimensions: ({m}, {n}, {k}), time"
        f" taken (median): {result.time_median * 1e3} ms, TFLOPS:"
        f" {ops / result.time_median / 1e12}"
    )

  pd.DataFrame(results).to_csv(OUTPUT, sep="\t", index=False)


if __name__ == "__main__":
  main()

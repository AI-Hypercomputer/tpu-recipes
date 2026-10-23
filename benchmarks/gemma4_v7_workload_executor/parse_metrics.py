# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parses InferenceX benchmark_serving.py output and writes TensorBoard events."""

import argparse
import json
import os
import re
import sys
import time

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

# Mapping from benchmark_serving.py JSON keys to BAP registry metric names.
JSON_KEY_MAP = {
    "request_throughput": "request_throughput",
    "output_throughput": "output_token_throughput",
    "total_token_throughput": "total_token_throughput",
    "mean_ttft_ms": "mean_ttft_ms",
    "median_ttft_ms": "median_ttft_ms",
    "p99_ttft_ms": "p99_ttft_ms",
    "mean_tpot_ms": "mean_tpot_ms",
    "median_tpot_ms": "median_tpot_ms",
    "p99_tpot_ms": "p99_tpot_ms",
    "mean_itl_ms": "mean_itl_ms",
    "median_itl_ms": "median_itl_ms",
    "p99_itl_ms": "p99_itl_ms",
}

# Fallback regex patterns matching standard benchmark_serving.py stdout output.
STDOUT_PATTERNS = {
    "request_throughput": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_token_throughput": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_token_throughput": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
}


def extract_metrics(result_json_path: str, client_log_path: str) -> dict[str, float]:
  metrics: dict[str, float] = {}

  if result_json_path and os.path.exists(result_json_path):
    with open(result_json_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    for src_key, dst_name in JSON_KEY_MAP.items():
      if src_key in data and data[src_key] is not None:
        metrics[dst_name] = float(data[src_key])

  if client_log_path and os.path.exists(client_log_path):
    with open(client_log_path, "r", encoding="utf-8") as f:
      log_text = f.read()
    for metric_name, pattern in STDOUT_PATTERNS.items():
      if metric_name not in metrics:
        match = re.search(pattern, log_text)
        if match:
          metrics[metric_name] = float(match.group(1))

  return metrics


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Convert InferenceX benchmark output to TensorBoard events."
  )
  parser.add_argument("--result-json", required=False, default="")
  parser.add_argument("--client-log", required=False, default="")
  parser.add_argument("--step", type=int, default=0)
  args = parser.parse_args()

  tblog_dir = os.environ.get("TENSORBOARD_OUTPUT_DIR")
  if not tblog_dir:
    print("::error::TENSORBOARD_OUTPUT_DIR environment variable is not set.", file=sys.stderr)
    sys.exit(1)

  metrics = extract_metrics(args.result_json, args.client_log)
  if not metrics:
    print("::error::No benchmark metrics found in JSON or client log.", file=sys.stderr)
    sys.exit(1)

  os.makedirs(tblog_dir, exist_ok=True)
  writer = EventFileWriter(tblog_dir)
  wall_time = time.time()

  for tag, value in metrics.items():
    event = event_pb2.Event(
        step=args.step,
        wall_time=wall_time,
        summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)]
        ),
    )
    writer.add_event(event)
    print(f"Logged metric {tag} = {value}")

  writer.close()
  print(f"Successfully wrote {len(metrics)} metrics to {tblog_dir}.")


if __name__ == "__main__":
  main()

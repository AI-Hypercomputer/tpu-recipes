# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Throughput benchmark for Qwen3-Reranker-0.6B on TPU via vLLM (torchax / tpu-inference).

Sweeps batch sizes and reports peak pairs/sec (query/document pairs scored per second)
using the pooling score() API. Also reports single-request (batch=1) latency. The model
is loaded from the original Qwen/Qwen3-Reranker-0.6B with the seq-cls head via hf_overrides.
"""

import argparse
import json
import os
import statistics
import time

from vllm import LLM

BQ = [
    "What is the capital of China?", "Explain gravity", "How do vaccines work?",
    "What causes rain?", "Who wrote Pride and Prejudice?", "What is photosynthesis?",
    "How does a transformer model work?", "What is the boiling point of water?",
]
BD = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies toward each other and gives weight to objects.",
    "Vaccines train the immune system by exposing it to a harmless piece of a pathogen.",
    "Rain forms when water vapor condenses into droplets heavy enough to fall.",
    "Pride and Prejudice was written by Jane Austen.",
    "Photosynthesis is how plants convert sunlight, water and CO2 into glucose and oxygen.",
    "A transformer uses self-attention to weigh the importance of tokens in a sequence.",
    "Water boils at 100 degrees Celsius at sea-level atmospheric pressure.",
]


def make(n):
    return [BQ[i % len(BQ)] for i in range(n)], [BD[i % len(BD)] for i in range(n)]


def pctl(s, p):
    if not s:
        return None
    k = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return round(s[k], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DEV_MODEL", "Qwen/Qwen3-Reranker-0.6B"))
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--batch-sizes", default="8,16,32,64,128,256,512,1024")
    ap.add_argument("--rt-iters", type=int, default=300)
    ap.add_argument("--result-filename", default="reranker-bench-tpu.json")
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        runner="pooling",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(8192, args.max_model_len * 16),
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    sizes = [int(x) for x in args.batch_sizes.split(",")]
    for bs in sizes:  # warmup shapes
        q, d = make(bs)
        try:
            llm.score(q, d)
        except Exception as e:  # noqa: BLE001
            print(f"warmup bs={bs} failed: {e}")

    rows, sample = [], None
    for bs in sizes:
        q, d = make(bs)
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            outs = llm.score(q, d)
            dt = time.perf_counter() - t0
            if sample is None:
                sample = [round(float(o.outputs.score), 4) for o in outs[:4]]
            tp = bs / dt if dt > 0 else 0
            if best is None or tp > best:
                best = tp
        rows.append({"batch_size": bs, "throughput_pairs_per_s": round(best, 3)})
        print(f"bs={bs}: {round(best, 3)} pairs/s")

    for i in range(10):
        llm.score([BQ[i % len(BQ)]], [BD[i % len(BD)]])
    lat = []
    for i in range(args.rt_iters):
        s0 = time.perf_counter()
        llm.score([BQ[i % len(BQ)]], [BD[i % len(BD)]])
        lat.append((time.perf_counter() - s0) * 1000.0)
    lat.sort()

    peak = max(rows, key=lambda r: r["throughput_pairs_per_s"])
    result = {
        "model": args.model,
        "device": "tpu v6e (torchax / tpu-inference)",
        "runner": "pooling (vLLM score API)",
        "max_model_len": args.max_model_len,
        "batch_throughput": rows,
        "max_throughput_pairs_per_s": peak["throughput_pairs_per_s"],
        "max_throughput_at_batch_size": peak["batch_size"],
        "realtime_bs1": {
            "iterations": args.rt_iters,
            "latency_ms_p50": pctl(lat, 50),
            "latency_ms_p99": pctl(lat, 99),
            "latency_ms_mean": round(statistics.mean(lat), 3),
        },
        "sample_scores": sample,
    }
    with open(args.result_filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"MAX {result['max_throughput_pairs_per_s']} pairs/s @bs "
          f"{result['max_throughput_at_batch_size']}; saved {args.result_filename}")


if __name__ == "__main__":
    main()

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

"""Offline reranking with Qwen3-Reranker-0.6B on TPU via vLLM (torchax / tpu-inference).

The public tpu-inference backend registers CausalLM-style architectures, so the
sequence-classification reranker is loaded from the original Qwen/Qwen3-Reranker-0.6B
weights with the seq-cls scoring head attached at load time via hf_overrides. Scoring
uses vLLM's pooling score() API: each (query, document) pair returns one relevance score.
"""

import json
import os

from vllm import LLM


def main():
    model = os.environ.get("DEV_MODEL", "Qwen/Qwen3-Reranker-0.6B")

    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=1024,
        max_num_batched_tokens=16384,
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    queries = [
        "What is the capital of China?",
        "Who wrote Pride and Prejudice?",
        "How do vaccines work?",
        "What is photosynthesis?",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Pride and Prejudice was written by Jane Austen.",
        "Vaccines train the immune system by exposing it to a harmless piece of a pathogen.",
        "Photosynthesis is how plants convert sunlight, water and CO2 into glucose and oxygen.",
    ]

    outputs = llm.score(queries, documents)
    report = {q: float(o.outputs.score) for q, o in zip(queries, outputs)}
    for q, s in report.items():
        print(f"{s:.4f}\t{q}")

    with open("rerank-output-tpu.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Results saved to rerank-output-tpu.json")


if __name__ == "__main__":
    main()

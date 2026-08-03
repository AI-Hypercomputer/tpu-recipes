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

import json
import os
import random
import numpy as np

from vllm import LLM


def main():
    default_model = "Qwen/Qwen3-Embedding-8B"
    model = os.environ.get("DEV_MODEL", default_model)

    target_device = os.environ.get("VLLM_TARGET_DEVICE", "cpu").lower()
    tp_size = 2 if target_device == "tpu" else 1

    llm = LLM(
        model=model,
        runner="pooling",
        max_num_seqs=16,
        max_model_len=16384,
        max_num_batched_tokens=512,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=tp_size
    )

    base_inputs = [
        "Hello, my name is Alice.",
        "In today's fast-paced world, finding a balance between productivity "
        "and mindfulness has become more important than ever. As urban "
        "landscapes continue to evolve, people are looking for ways to "
        "reconnect with nature without losing the convenience of modern "
        "technology.",
        "最近の技術革新により、私たちの日常生活は劇的に変化しました。"
        "都市の風景は新旧の建築が入り混じり、静かな朝の光が窓から差し込む中で、"
        "人々はそれぞれの目的を持って歩き始めます。",
        "최근 기술의 발전과 함께 우리의 일상에는 많은 변화가 찾아왔습니다.",
    ]
    inputs = [text * 180 for text in base_inputs]

    results = llm.embed(inputs)

    report = dict(zip(base_inputs, [r.outputs.embedding for r in results]))

    output_file = "embed-output-tpu.json" if os.environ.get("VLLM_TARGET_DEVICE") == "tpu" else "embed-output-cpu.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

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
import numpy as np
import torch
import multiprocessing as mp

# Select Mode: 'text'  or 'multimodal'
test_mode = os.environ.get("TEST_MODE", "text").lower()
target_device = os.environ.get("VLLM_TARGET_DEVICE", "cuda").lower()

os.environ["MODEL_IMPL_TYPE"] = "vllm"

if target_device == "tpu":
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
elif target_device == "cuda":
    os.environ["VLLM_TARGET_DEVICE"] = "cuda"

def main():
    from vllm import LLM
    from vllm.multimodal.utils import fetch_image

    model_name = "Qwen/Qwen3-VL-Embedding-8B"
    
    # TPU uses TP=8
    # GPU uses TP=1
    tp_size = 8 if target_device == "tpu" else 1
    
    max_tokens = 13312

    print(f"Device: {target_device} | Mode: {test_mode} | TP: {tp_size}")

    # Initialize LLM with pooling runner
    llm = LLM(
        model=model_name,
        runner="pooling",
        max_model_len=16384,
        max_num_batched_tokens=max_tokens,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.7
    )

    if test_mode == "multimodal":
        # Mode A: Full multimodal test
        image_placeholder = "<|image_pad|>"
        image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
        image = fetch_image(image_url)
        
        # Total tokens ~15,600 to force StepPool
        prompt = f"{image_placeholder}\nPlease analyze this image: " + "Test text. " * 1200
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}}]
    else:
        # Mode B: Pure text test to isolate the Backbone and StepPool logic.
        # ~15,000 tokens to ensure StepPool is triggered
        prompt = "Backbone alignment test block. " * 1500 
        inputs = [{"prompt": prompt}]

    print(f"Executing {target_device} inference...")
    results = llm.embed(inputs)

    # Normalize output format for comparison
    report = {"embedding": results[0].outputs.embedding}
    output_file = f"embed-{test_mode}-{target_device}.json"
    
    with open(output_file, "w") as f:
        json.dump(report, f)

    print(f"Success! Results saved to {output_file}")

if __name__ == "__main__":
    main()

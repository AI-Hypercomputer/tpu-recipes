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
import torch
import torch.nn.functional as F
import os

def load_emb(path):
    """Load embedding tensor from the JSON report file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
        # Standardize key access based on the latest report format
        if "embedding" in data:
            return torch.tensor(data["embedding"])
        elif "multimodal_step_pooling" in data:
            return torch.tensor(data["multimodal_step_pooling"])
        return None

def display_parity(mode, tpu_path, gpu_path):
    """Calculate and display cosine similarity for a device pair."""
    tpu_emb = load_emb(tpu_path)
    gpu_emb = load_emb(gpu_path)
    
    print(f"\n--- Result for [{mode.upper()}] Mode ---")
    if tpu_emb is None or gpu_emb is None:
        print(f"File status: Missing required data ({tpu_path} or {gpu_path})")
        return

    # Cosine Similarity Calculation
    # similarity = (A . B) / (||A|| ||B||)
    sim = F.cosine_similarity(tpu_emb.unsqueeze(0), gpu_emb.unsqueeze(0)).item()
    
    print(f"TPU File: {tpu_path}")
    print(f"GPU File: {gpu_path}")
    print(f"Similarity: {sim:.8f}")

def main():
    print("Numerical Alignment Summary (TPU vs GPU)")
    print("==================================================")
    
    display_parity("text", "embed-text-tpu.json", "embed-text-cuda.json")
    
    display_parity("multimodal", "embed-multimodal-tpu.json", "embed-multimodal-cuda.json")

if __name__ == "__main__":
    main()

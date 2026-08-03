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
import numpy as np
import datetime

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    with open("embed-output-cpu.json", "r") as f:
        cpu_data = json.load(f)
    with open("embed-output-tpu.json", "r") as f:
        tpu_data = json.load(f)

    output_str = f"{'Text Content (Prefix)':<40} {'Cosine Similarity':<20}\n"
    output_str += "-" * 60 + "\n"

    for key in cpu_data:
        v1 = cpu_data[key]
        v2 = tpu_data.get(key)
        if v2 is None:
            output_str += f"{key[:37]+'...':<40} {'Not found in TPU':<20}\n"
            continue
        
        sim = cosine_similarity(v1, v2)
        output_str += f"{key[:37]+'...':<40} {sim:.10f}\n"

    print(output_str, end="")
    
    # Automatically log to file
    with open("precision_results.log", "a") as f:
        f.write(f"--- Run at {datetime.datetime.now()} ---\n")
        f.write(output_str)
        f.write("\n")

if __name__ == '__main__':
    main()

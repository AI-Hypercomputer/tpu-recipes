# Serve vLLM on Trillium TPUs (v6e)

This repository provides examples demonstrating how to deploy and serve vLLM on Trillium TPUs using GCE (Google Compute Engine) for a select set of models.

- [Llama3.1-8B/70B](./Llama3.1/README.md)
- [Qwen2.5-32B](./Qwen2.5-32B/README.md)
- [Qwen2.5-VL-7B](./Qwen2.5-VL/README.md)
- [Qwen3-4B/32B](./Qwen3/README.md)

These models were chosen for demonstration purposes only. You can serve any model from this list: [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

If you are looking for GKE-based deployment, please refer to this documentation: [Serve an LLM using TPU Trillium on GKE with vLLM](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)

## Choosing the Right TPU Configuration

Selecting the appropriate TPU size is critical for performance and cost-effectiveness. The goal is to use the smallest TPU configuration that can accommodate the model's memory requirements. These recommendations assume the model is running in a standard 16-bit precision format like bfloat16 or float16.

*   **✅ Recommended:** The most cost-effective configuration.
*   **⚠️ Overkill:** The model will run, but the TPU is larger and more expensive than necessary.
*   **❌ Insufficient Memory:** The model will not fit in the TPU's memory.

| Model | v6e-1 (32 GB) | v6e-4 (128 GB) | v6e-8 (256 GB) |
| :---- | :---: | :---: | :---: |
| **Qwen3-4B** | ✅ | ⚠️ | ⚠️ |
| **Qwen2.5-VL-7B**| ✅ | ⚠️ | ⚠️ |
| **Llama3.1-8B** | ✅ | ⚠️ | ⚠️ |
| **Qwen2.5-32B** | ❌ | ✅ | ⚠️ |
| **Qwen3-32B** | ❌ | ✅ | ⚠️ |
| **Llama3.1-70B**| ❌ | ❌ | ✅ |

**Note on Topology:** The topology (e.g., `2x2` for 4 chips, `2x4` for 8 chips) describes the physical arrangement of the TPU chips. This layout affects the communication speed between chips. While any valid topology with the correct number of chips will work, a more compact topology (like `2x2` vs. `1x4`) can reduce latency and improve performance for communication-heavy models. For general use, the default topology is usually sufficient, but performance-critical applications may benefit from tuning this setting.

**Note on Availability:** Acquiring on-demand TPUs can be challenging due to high demand. If you encounter capacity limits in one zone, we recommend trying a different zone or using [Queued Resources](https://cloud.google.com/tpu/docs/queued-resources) to ensure you get the required capacity.

# Serve vLLM on Trillium TPUs (v6e)

Although vLLM TPU’s [new unified backend](https://github.com/vllm-project/tpu-inference) makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.

For this reason, we’ve provided a set of stress-tested recipes for deploying and serving vLLM on Trillium TPUs using Google Compute Engine (GCE).

- [Llama3.1-8B/70B](./Llama3.1/README.md)
- [Llama3.3-70B](./Llama3.3/README.md)
- [Qwen2.5-32B](./Qwen2.5-32B/README.md)
- [Qwen2.5-VL-7B](./Qwen2.5-VL/README.md)
- [Qwen3-4B/32B](./Qwen3/README.md)

If you are looking for GKE-based deployment, please refer to this documentation: [Serve an LLM using TPU Trillium on GKE with vLLM](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)

Please consult the [Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features.html) page for a list of models and features that are validated through unit, integration, and performance testing.

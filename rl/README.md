# Reinforcement-learning (post-training) recipes

Recipes in this folder reproduce step-time measurements of reinforcement-learning
post-training on Cloud TPU. They follow the layout of `../training` and
`../inference`: `rl/<tpu generation>/<model>-<algorithm>/<workload>/<launcher>/`.

Unlike a pretraining step, an RL step is a mixed workload: it generates
completions with an inference engine, scores them with a reward function, and
then trains on the scored samples. Step time is reported for the whole loop.
Tokens/sec/chip counts generated tokens; TFLOPs/sec/chip is not reported.

## Ironwood

| Model ID | Algorithm | Framework | Chips | Prompts x samples | Prompt + completion cap | Precision | Step time (seconds) | Tokens/sec/chip (generated) |
|---|---|---|---|---|---|---|---|---|
| [qwen3-0.6b](ironwood/qwen3-0.6b-grpo/math-bf16-tpu7x-4x4x4/k8s/README.md) | GRPO | MaxText + Tunix + vLLM TPU | 64 (4x4x4) | 256 x 8 | 4,096 + 8,192 | bf16 | 56.94 | 2,491.9 |

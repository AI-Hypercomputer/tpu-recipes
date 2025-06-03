# Cloud TPU performance recipes

This repository provides the necessary instructions to reproduce a
specific workload on Google Cloud TPUs. The focus is on reliably achieving
a performance metric (e.g. throughput) that demonstrates the combined hardware
and software stack on TPUs.

## Organization

- `./training`: instructions to reproduce the training performance of
  popular LLMs, diffusion, and other models with PyTorch and JAX.

- `./inference`: instructions to reproduce inference performance.

- `./microbenchmarks`: instructions for low-level TPU benchmarks such as
  matrix multiplication performance and memory bandwidth.

## Contributor notes

Note: This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Hello!
# Cloud TPU performance recipes

This repository provides the instructions necessary to reproduce specific
workload performance measurements, which are part of a confidential
benchmarking program. This repository focuses on helping you reliably
achieve performance metrics, for example, throughput that demonstrates
the combined hardware and software stack on TPUs.

**Note:** The content in this repository is not designed as a set of
general-purpose code samples or tutorials for using Compute Engine-based products.

## Intended audience

This content is for you if you are a customer or partner who needs to:

- validate hardware performance with your suppliers
- inform purchasing decisions using the benchmarking data
- reproduce optimal performance scenarios before you customize workflows
  for your own requirements

## How to use these recipes

To reproduce a benchmark, follow these steps:

- **Identify your requirements:** determine the model, TPU type, workload, and
  framework (JAX or PyTorch) you are interested in.
- **Select a recipe:** navigate to the appropriate directory, for example,
  `./training` or `./inference`, to find a recipe that meets your needs.
- **Follow the procedure:** Each recipe guides you through preparing your
  environment, running the benchmark, and analyzing the results (including detailed logs).

## Repository organization

- `./training`: Use these instructions to reproduce the training performance of
  popular LLMs, diffusion, and other models with PyTorch and JAX.
- `./inference`: Use these instructions to reproduce inference performance.
- `./microbenchmarks`: Use these instructions for low-level TPU benchmarks
  such as matrix multiplication performance and memory bandwidth.
- `./utils`: Find utility scripts here for your cluster and resource
  management, for example, Ironwood for GKE TPU v7.

## Repository scope

This repository provides the steps that you can use to reproduce a specific
benchmark. The actual performance measurements or the complete, confidential
benchmark report are not included.

## Methodology

Performance benchmarks measure the performance of various workloads on the 
platform. These benchmarks are primarily used to validate performance with
hardware suppliers and to provide you with data for purchasing decisions.

### Maintenance policy

Benchmark data is considered a point-in-time measurement and completed
benchmarks are not repeated. As such, there is no intent to maintain or
update the reproducibility steps provided in this repository.

## Resources

If you are looking for general guidance on how to get started using
Compute products, refer to the official documentation and tutorials:

- [Official Compute Engine tutorials and samples](https://docs.cloud.google.com/compute/docs/overview)
- [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs)
- [AI Hypercomputer documentation](https://docs.cloud.google.com/ai-hypercomputer/docs)

## Getting help

If you have any questions or if you encounter any problems with this repository,
report them through https://github.com/AI-Hypercomputer/tpu-recipes/issues.

## Contributor notes

Note: This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

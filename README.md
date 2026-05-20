# Cloud TPU performance benchmark recipes

This repository contains recipes that provide instructions to reproduce specific
workload performance measurements, which are part of a confidential benchmarking
program. These recipes focus on helping you reliably achieve performance metrics,
such as throughput, that demonstrate the combined hardware and software stack on
TPUs.

**Note:** The recipes in this repository are not designed as general-purpose code
samples or tutorials for using Compute Engine-based products.

## Intended audience

This content is for you if you are a customer or partner who needs to:
- Validate hardware performance with your suppliers.
- Inform purchasing decisions using the benchmarking data.
- Reproduce optimal performance scenarios before you customize workflows for your
  own requirements.

## How to use these recipes

To reproduce a benchmark, follow these steps:

1.**Identify your requirements:** determine the model, TPU version, workload, and
  framework (JAX or PyTorch) that you are interested in.
2.**Select a recipe:** navigate to the appropriate directory, such as `./training`
  or `./inference`, to find a recipe that meets your needs.
3.**Follow the procedure:** each recipe guides you through preparing your environment,
  running the benchmark, and analyzing the results (including detailed logs). You can
  automate your infrastructure setup using Cluster Toolkit. For more information, see
  [Automated TPU environment deployment with Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview). 

## Repository organization

- `./training`: This directory contains recipes with instructions to reproduce the
  training performance of popular models, using PyTorch and JAX on specific TPU versions.
- `./inference`: This directory contains recipes that provide instructions and
  configurations to reproduce inference performance of models running on specific TPU
  versions.
- `./microbenchmarks`: This directory contains instructions for running low-level
  performance tests on TPUs, specifically focusing on matrix multiplication
  performance and memory bandwidth.
- `./utils`: This directory contains utility scripts for cluster and resource management
  for TPU7x (Ironwood) in GKE. For fully automated, production-ready cluster deployment,
  we recommend using the [Automated TPU environment deployment with Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x).


## Repository scope

This repository provides the steps that you can use to reproduce a specific benchmark. 
The actual performance measurements and the complete, confidential benchmark report are 
not included.

## Methodology

Performance benchmarks measure the performance of various workloads on the platform. 
These benchmarks are primarily used to validate performance with hardware suppliers and 
to provide you with data for purchasing decisions.

## Maintenance policy

Benchmark data is considered a point-in-time measurement and completed benchmarks are not 
repeated. We maintain and update the recipes in this repository on a best-effort basis.

## Resources

For general guidance on using Google Cloud compute products, see the official documentation
and tutorials:

- [Official Compute Engine tutorials](https://docs.cloud.google.com/compute/docs/overview)
- [Official Compute Engine samples](https://docs.cloud.google.com/compute/docs/samples)
- [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs)
- [AI Hypercomputer documentation](https://docs.cloud.google.com/ai-hypercomputer/docs)
- [Automated TPU environment deployment with Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview)

## Report issues

If you have questions or encounter problems with this repository, report them through
[GitHub Issues](https://github.com/AI-Hypercomputer/tpu-recipes/issues) or reach out to
your Google Cloud account team for assistance.

## Contributor notes

Note: This is not an officially supported Google product. This project is not eligible for
the  [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

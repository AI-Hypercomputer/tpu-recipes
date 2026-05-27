# HBM Microbnechmarks on tpu7x-2x2x1

This guide provides instructions for running High Bandwidth Memory (HBM) microbenchmarks on tpu7x-2x2x1 Google Kubernetes Engine (GKE) clusters. It covers creating a node pool, running the benchmarks, and viewing the expected output.

## Create Node Pools

Follow [Setup section](../../Ironwood_Microbenchmarks_readme.md#setup) to create a GKE cluster with one 2x2x1 nodepool.

## Run HBM Microbenchmarks

To run the HBM microbenchmarks, apply the following Kubernetes configuration:
```bash
kubectl apply -f tpu7x-2x2x1-hbm-microbenchmark.yaml
```

To extract the log of HBM microbenchmark, use `kubectl log`:
```bash
kubectl log tpu7x-2x2x1-hbm-microbenchmark
```

Once the benchmark completes, you should see logs similar to the example below:

```bash
Tensor size: 8192.0 MB, time taken (median): 5.3523 ms, bandwidth (median): 3209.812 GB/s

Writing metrics to JSONL file: ../microbenchmarks/hbm/metrics_report.jsonl
Metrics written to CSV at ../microbenchmarks/hbm/t_single_device_hbm_copy_[A-Z0-9]+.tsv.
```

To retrieve the complete results, including the trace and TSV output files, you must keep the pod running after the benchmark completes. To do this, add a `sleep` command to the `tpu7x-2x2x1-hbm-microbenchmark.yaml` file. You can then use `kubectl cp` to copy the output from the pod.

```bash
kubectl cp tpu7x-2x2x1-hbm-microbenchmark:/microbenchmarks/hbm hbm
```

## Expected bandwidth for different matrix size


| Matrix Size (Bytes) | Bandwidth (GB/s/core) | Bandwidth (GB/s/chip) |
|---------------------|-----------------------|-----------------------|
|             2097152 |           1379.335021 |           2758.670041 |
|             4194304 |           2249.746091 |           4499.492181 |
|             8388608 |           2246.129937 |           4492.259875 |
|            16777216 |           2757.308985 |            5514.61797 |
|            33554432 |            3009.83593 |            6019.67186 |
|            67108864 |           3097.217778 |           6194.435556 |
|           134217728 |            3176.50274 |           6353.005481 |
|           268435456 |           3167.144485 |           6334.288969 |
|           536870912 |           3199.020504 |           6398.041009 |
|          1073741824 |           3198.414211 |           6396.828421 |
|          2147483648 |           3203.486119 |           6406.972238 |
|          4294967296 |           3197.879607 |           6395.759214 |
|          8589934592 |           3210.480912 |           6420.961823 |
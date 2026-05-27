# All Gather Microbenchmark

This guide demonstrates how to performance collective benchmark for the `2x2x1` and `2x2x2` topologies.

## Setup
Please follow the prerequisites setup [here](../../Ironwood_Microbenchmarks_readme.md#prerequisites).

### Create Node Pool

Refer to the instructions [here](../../Ironwood_Microbenchmarks_readme.md#setup) to create `2x2x1` and `2x2x2` node pools, making sure to adjust the accelerator-topology from `4x4x4` to `2x2x2`.

## Run Benchmarks

Deploy the all_gather microbenchmarks for different TPU topologies.
```
kubectl apply -f tpu7x-2x2x1-ici-all-gather-microbenchmark.yaml
kubectl apply -f tpu7x-2x2x2-ici-all-gather-microbenchmark.yaml
```

### Monitor Results

Use the following commands to retrieve the benchmark logs.
```bash
kubectl logs pod/tpu7x-2x2x1-ici-all-gather-microbenchmark
kubectl logs job.batch/tpu7x-2x2x2-ici-all-gather-microbenchmark
```

Once the benchmark completes, you should see logs similar to the example below:
```bash
metadata:  {'iteration': 16384, 'op_type': 'AG' ... }
metrics:  {... 'achieved_bw (GB/s)_max': np.float64(159.35667369827922) ...}
Writing metrics to JSONL file: ../microbenchmarks/all_gather/metrics_report.jsonl
Metrics written to CSV at ../microbenchmarks/all_gather/t_all_gather_[A-Z0-9]+.tsv.
```

To retrieve the complete results, use the `kubectl cp` command to copy the TSV report file, typically named `t_all_gather_[A-Z0-9]+.tsv`, from the `/microbenchmarks` directory within the pod.

### Cleanup

```bash
kubectl delete -f tpu7x-2x2x1-ici-all-gather-microbenchmark.yaml
kubectl delete -f tpu7x-2x2x2-ici-all-gather-microbenchmark.yaml
```

## Expected Results
| Topology | Number of Elements | Achieved Bandwidth (GB/s) | Transferred Data (GB) | Input Shape      | Output  Shape    |
| -------- | ------------------ | ------------------------- | --------------------- | ---------------- | ---------------- |
| 2x2x1    | 65536              | 71.7703034                | 0.001572864           | f32[64,8,128]    | f32[256,8,128]   |
| 2x2x1    | 262144             | 129.1869148               | 0.006291456           | f32[256,8,128]   | f32[1024,8,128]  |
| 2x2x1    | 1048576            | 161.9040742               | 0.025165824           | f32[1024,8,128]  | f32[4096,8,128]  |
| 2x2x1    | 4194304            | 174.765465                | 0.100663296           | f32[4096,8,128]  | f32[16384,8,128] |
| 2x2x1    | 16777216           | 178.7714158               | 0.402653184           | f32[16384,8,128] | f32[65536,8,128] |
| 2x2x2    | 65536              | 69.53210305               | 0.001572864           | f32[64,8,128]    | f32[256,8,128]   |
| 2x2x2    | 262144             | 127.3945462               | 0.006291456           | f32[256,8,128]   | f32[1024,8,128]  |
| 2x2x2    | 1048576            | 162.0864029               | 0.025165824           | f32[1024,8,128]  | f32[4096,8,128]  |
| 2x2x2    | 4194304            | 174.6652133               | 0.100663296           | f32[4096,8,128]  | f32[16384,8,128] |
| 2x2x2    | 16777216           | 178.7592252               | 0.402653184           | f32[16384,8,128] | f32[65536,8,128] |

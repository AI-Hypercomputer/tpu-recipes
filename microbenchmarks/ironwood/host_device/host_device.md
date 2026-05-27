# Host Device Microbenchmarks on tpu7x-2x2x1

This guide provides instructions for running Host Device (Host-to-Device and Device-to-Host) microbenchmarks on tpu7x-2x2x1 Google Kubernetes Engine (GKE) clusters. It covers creating a node pool, running the benchmarks, and viewing the output.

> [!WARNING]
> This benchmark is currently a Work In Progress (WIP). Expected bandwidth numbers are not yet finalized.

## Create Node Pools

Follow [Setup section](../../Ironwood_Microbenchmarks_readme.md#setup) to create a GKE cluster with one 2x2x1 nodepool.

## Run Host Device Microbenchmarks

To run the microbenchmarks, apply the following Kubernetes configuration:
```bash
kubectl apply -f tpu7x-host-device-benchmark.yaml
```

To extract the log of the microbenchmark, use `kubectl logs`:
```bash
kubectl logs tpu7x-host-device-benchmark
```

Once the benchmark completes, you should see logs reporting bandwidth statistics.

To retrieve the complete results, including the trace and CSV output files, you must keep the pod running after the benchmark completes. To do this, add a `sleep` command to the `tpu7x-host-device-benchmark.yaml` file. You can then use `kubectl cp` to copy the output from the pod.

```bash
kubectl cp tpu7x-host-device-benchmark:/microbenchmarks/host_device host_device
```

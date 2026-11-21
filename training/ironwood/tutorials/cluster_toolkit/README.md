# Ironwood (TPU v7x) end-to-end tutorials with Cluster Toolkit

These tutorials show how to deploy an Ironwood (Cloud TPU v7x) GKE cluster and
its backing storage declaratively with
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`) using the `examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml` blueprint,
and how to submit a training workload with `gcluster job submit`. Each tutorial follows the same cluster + storage → workload → cleanup flow.

| Tutorial | Capacity | Storage |
| --- | --- | --- |
| [Reservation + Cloud Storage](reservation-gcs/README.md) | Regular, gSC, or DWS Calendar reservation | Cloud Storage buckets + GCS FUSE Storage Profile PVCs (`gcsfusecsi-checkpointing` / `gcsfusecsi-training`) |
| [Flex-start + Filestore](flex-start-filestore/README.md) | DWS flex-start (node pool scales from 0) | Filestore (`modules/file-system/filestore`) + `ReadWriteMany` PVC |
| [Flex-start + Managed Lustre](flex-start-lustre/README.md) | DWS flex-start (node pool scales from 0) | Managed Lustre (`modules/file-system/managed-lustre`) + PSA + `ReadWriteMany` PVC |

For model-specific recipes on Ironwood (Llama 3.1, DeepSeek-V3, GPT-OSS, Qwen3,
Gemma 4, and others), see the `cluster_toolkit/` directories under
[`training/ironwood/`](../../).

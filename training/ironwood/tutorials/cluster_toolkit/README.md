# Ironwood (TPU v7x) end-to-end tutorials with Cluster Toolkit

These tutorials show how to create an Ironwood (Cloud TPU v7x) GKE cluster with
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`), attach storage, and run a training workload. Each one follows the
same cluster → storage → workload structure and is the Cluster Toolkit
equivalent of the corresponding
[XPK TPU 7x recipe](https://github.com/AI-Hypercomputer/xpk/tree/main/docs/usage/tpu7x/recipes).

| Tutorial | Capacity | Storage |
| --- | --- | --- |
| [Reservation + Cloud Storage](reservation-gcs/README.md) | Regular, gSC, or DWS Calendar reservation | Cloud Storage bucket via GCS FUSE CSI |
| [Flex-start + Filestore](flex-start-filestore/README.md) | DWS flex-start (node pool scales from 0) | Filestore via Filestore CSI |
| [Flex-start + Managed Lustre](flex-start-lustre/README.md) | DWS flex-start (node pool scales from 0) | Managed Lustre via Lustre CSI |

For model-specific recipes on Ironwood (Llama 3.1, DeepSeek-V3, GPT-OSS, Qwen3,
Gemma 4, and others), see the `cluster_toolkit/` directories under
[`training/ironwood/`](../../).

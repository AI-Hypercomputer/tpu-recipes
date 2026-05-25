# Instructions for training Llama3.1-8B-MaxText on TPU trillium (v6e-8)

## GCluster setup
Please set up the cluster and download the `gcluster` tool. You should configure your environment variables as described below.

> [!TIP]
> For a detailed, comprehensive guide on the GCluster workload submission framework, dynamic configuration overrides, and advanced scheduling capabilities, refer to the official [GCluster Job Guide](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/docs/gcluster_job_guide.md).


## Prep for Maxtext

### Install MaxText and Build Docker Image
Please install MaxText and build a Docker image pushed to Google Artifact Registry or container registry. 

Ensure the following variables are set in your environment:
```bash
export CLUSTER_NAME="" # The name of your GKE cluster
export PROJECT=""       # Your GCP project ID
export LOCATION=""      # The location/zone/region of your GKE cluster
export IMAGE_NAME=""    # Your pre-built container image URL
export OUTPUT_DIR=""    # GCS bucket path (e.g. gs://my-bucket/maxtext_output)

# Optional overrides:
export COMPUTE_TYPE=""  # GKE TPU node type (defaults to "v6e-4")
export TOPOLOGY=""      # Slice topology (defaults to "2x4")
export QUEUE=""         # Kueue LocalQueue (defaults to "multislice-queue")
```


## Run Maxtext Llama3.1-8B workloads on GKE

### 1. Optional: Setting up Kueue Queues (First-Time Setup Only)

If you are running on a freshly deployed GKE cluster, or if Kueue is installed but its resource queues have not been configured, the submitted jobs will hang in a `Pending` (Not Admitted) state. 

You can configure the required topology-aware Kueue queues by executing:
```bash
kubectl apply -f kueue-resources.yaml
```
*This provisions a `ResourceFlavor`, `ClusterQueue`, and `LocalQueue` in your namespace configured specifically to admit and schedule this TPU topology workload.*

### 2. Starting Workload

From the recipe directory, run the `gcluster` workload submission script:

```bash
bash llama3-1-8B-1xv6e-8-gcluster.sh
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 14, seconds: 3.443, TFLOP/s/device: 413.433, Tokens/s/device: 7138.890, total_weights: 196608, loss: 2.136
```

### Workload Details

For reference, here are the `llama3_1_8b_8192_no_collective_matmul` workload details as found in `MaxText@tpu-recipes-v0.1.4`:

```
MaxTextModel(
    model_name="llama3_1-8b-8192-no-collective-matmul",
    model_type="llama3.1-8b",
    tuning_params={
        "per_device_batch_size": 3,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "out_proj": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "enable_checkpointing": False,
        "sa_block_q": 2048,
        "sa_block_kv": 2048,
        "sa_block_kv_compute": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 2048,
        "sa_block_q_dq": 2048,
        "sa_block_kv_dq": 2048,
        "sa_use_fused_bwd_kernel": True,
        "profiler": "xplane",
        "skip_first_n_steps_for_profiler": 10,
        "profiler_steps": 5,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.DISABLE_COLLECTIVE_MATMUL
    ),
)
```

This equivalent workload code can be found in the [maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/9f1820b472ef362e7b5c782fe1d6fda8a0943eff/benchmarks/maxtext_trillium_model_configs.py) file within the MaxText repository.

## Troubleshooting & Operations

This section lists common errors and commands to manage and troubleshoot your GCluster workloads.

### 1. Useful Operations Commands

* **List submitted workloads:**
  ```bash
  gcluster job list --cluster "$CLUSTER_NAME" --location "$LOCATION" --project "$PROJECT"
  ```
* **View workload logs:**
  ```bash
  gcluster job logs [workload-name] --cluster "$CLUSTER_NAME" --location "$LOCATION" --project "$PROJECT"
  ```
  *(Or use `kubectl logs -f [pod-name]` for native Kubernetes logging)*
* **Cancel/delete a workload:**
  ```bash
  gcluster job cancel [workload-name] --cluster "$CLUSTER_NAME" --location "$LOCATION" --project "$PROJECT"
  ```

---

### 2. Common Errors & Solutions

#### A. PriorityClass Update Failure (`Forbidden: may not be changed in an update`)
If you upgrade Kueue or re-install it, you may see the following error:
```
Error: kubectl apply failed with exit code 1: PriorityClass.scheduling.k8s.io "very-low" is invalid: value: Forbidden: may not be changed in an update.
```
* **Cause:** GKE `PriorityClass` values are immutable once created, and updating them triggers an API server rejection.
* **Solution:** Manually delete the conflicting classes and re-run submission:
  ```bash
  kubectl delete priorityclass very-low low medium high very-high
  ```

#### B. Compile-Time Out-of-Memory (`CompileTimeHbmOom` / `RESOURCE_EXHAUSTED`)
When running on smaller TPU node slices (like `v6e-4` with a `2x2` topology) with the default Llama 3.1 8B settings, you may see:
```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: E1000: CompileTimeHbmOom: XLA:TPU compile permanent error. Ran out of memory in memory space hbm.
```
* **Cause:** Storing parameters and large context attention activations exceeds the physical 32 GB HBM memory on individual v6e chips.
* **Solution:** Reduce the memory footprint in the training command arguments within `llama3-1-8B-1xv6e-8-gcluster.sh`:
  1. Change `per_device_batch_size` from `3` to **`1`** (Line 56).
  2. Reduce `max_target_length` from `8192` to **`2048`** or **`4096`** (Line 63).

#### C. Artifact Registry Pull Access Denied (`ErrImagePull` / `403 Forbidden`)
If your pods are admitted but remain in `Pending` with an `ErrImagePull` status:
```
failed to authorize: failed to fetch oauth token: unexpected status from GET request: 403 Forbidden
```
* **Cause:** GKE's node pool Service Account does not have IAM permission to read from your custom Google Artifact Registry repository.
* **Solution:** Find your node pool service account name (typically `[node-pool-prefix]-gke-np-sa@...` or Compute Engine default) and grant the reader role:
  ```bash
  gcloud artifacts repositories add-iam-policy-binding [repo-name] \
      --location=[repo-region] \
      --member="serviceAccount:[your-gke-node-sa]@[project-id].iam.gserviceaccount.com" \
      --role="roles/artifactregistry.reader" \
      --project=[project-id]
  ```

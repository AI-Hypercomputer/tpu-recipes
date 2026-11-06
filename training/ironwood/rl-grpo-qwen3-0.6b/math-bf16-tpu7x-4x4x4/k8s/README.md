# GRPO reinforcement learning of Qwen3-0.6B on Ironwood GKE clusters with Kubernetes JobSet

This recipe outlines the steps for running a Qwen3-0.6B
[GRPO](https://arxiv.org/abs/2402.03300) reinforcement-learning workload with
[MaxText](https://github.com/AI-Hypercomputer/maxtext),
[Tunix](https://github.com/google/tunix) and
[vLLM TPU](https://github.com/vllm-project/tpu-inference) on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by applying a
Kubernetes manifest to deploy a JobSet resource. The job is orchestrated by
[Pathways](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro):
one head pod drives the loop and 16 worker pods hold the TPU slice.

## Workload details

This workload is configured with the following details:

-   Model: Qwen3-0.6B (bfloat16), trainer and sampler colocated on the same chips
-   Task: GRPO on the `train_1M` split of
    [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2);
    reward = answer correctness plus format, no KL term
-   Batch: 256 prompts x 8 generations = 2,048 completions per step
-   Lengths: prompt up to 4,096 tokens, completion up to 8,192 tokens
-   Sampling: temperature 0.8, top-k 50, top-p 0.95; rollout layout TP 1 x DP 128
-   Optimizer: learning rate 1e-6, clip 0.2 (low) / 0.28 (high), 20 steps
-   Chips: 64 (4x4x4 topology; 16 hosts x 4 chips = 128 TensorCores)

### Reference result

| Metric | Value |
|---|---|
| `perf/train/global_step_time`, median over TensorBoard steps 3..20 (n = 18) | **56.94 s** |
| p10 / p90 of those 18 steps | 55.2 s / 59.3 s |
| Step 1 (includes compilation) | about 160 s |
| Mean completion length, steps 3..20 | 4,150 to 4,750 tokens (about 29% of completions reach the 8,192 cap) |

The step time is the full step: generation, reward scoring, training and the
trainer-to-sampler weight sync. The number comes from the TensorBoard event file,
not from the `PHASETIME` lines in the pod log (those exclude the weight sync).

## Prerequisites

This recipe assumes the following prerequisites are met:

-   **GKE cluster:** A GKE cluster with a `tpu7x` node pool of topology `4x4x4`
    (machine type `tpu7x-standard-4t`, 16 nodes), a CPU node pool for the
    Pathways head pod (at least 44 vCPU and 170 GiB memory per node), and
    [JobSet](https://jobset.sigs.k8s.io/docs/installation/) installed. See
    [create a GKE cluster with Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster).
-   **Container image:** The MaxText post-training image built as described below,
    pushed to a registry the cluster can pull from.
-   **Checkpoint:** The Qwen3-0.6B weights converted to MaxText format, in a GCS
    bucket the cluster's nodes can read (see below).
-   **GCS bucket:** A bucket in the same region as the cluster for outputs and
    compilation caches.
-   **Tools:** `gcloud`, `kubectl`, `gke-gcloud-auth-plugin`, `envsubst`, `docker`
    and `python3` on your workstation. If `envsubst` is missing, install it with
    `sudo apt-get update && sudo apt-get install -y gettext-base`.
-   **Permissions:** You can run `kubectl apply` on the target cluster, and the
    cluster's node service account can pull the image and read/write the bucket.

## Orchestration and deployment tools

For this recipe, the following setup is used:

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
    with [Pathways](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)
-   **Job configuration and deployment** - A Kubernetes manifest
    (`k8s_manifest.yaml`) is used to define and deploy the
    [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource. It contains one ConfigMap (the launcher script with the full
    MaxText command line) and one JobSet (the Pathways head and 16 workers).

## Build the container image

The image is MaxText at a fixed commit with its three post-training dependencies
(Tunix, tpu-inference, vLLM) pinned to the commits the reference run used. No source
patches are applied. **Moving the three pins is required:** MaxText's own pin file
points at older commits, and an image built without this step starts and trains
normally but runs about twice as slow, because MaxText silently ignores the
configuration keys the older Tunix does not know.

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout c28166ca7de7f3b3cbca34ea3989fe50bb8c8c46

cat > src/dependencies/extra_deps/post_train_github_deps.txt <<'PINS'
google-tunix @ https://github.com/google/tunix/archive/a3e92fbaef146cb0f14c1dc95d92e353f8fe1fab.zip
tpu-inference @ https://github.com/vllm-project/tpu-inference/archive/c2289ee4bc7c7096a259bda5102319a13aaf496a.zip
vllm @ git+https://github.com/vllm-project/vllm@51da0ca66c8065619c79e35dff97aa99aeaf5644
PINS
git diff --stat   # expected: 1 file changed, 3 insertions(+), 3 deletions(-)

uv venv --python 3.12 venv && source venv/bin/activate
uv pip install -e .[runner] --resolution=lowest
build_maxtext_docker_image WORKFLOW=post-training
```

The build needs Docker with BuildKit, about 60 GB of free disk, and network access
to github.com and pypi.org. It produces a local image named `maxtext_base_image`.
Do not edit any other file; in particular do not change the generated requirements
(they resolve to jax/jaxlib 0.11.0 and libtpu 0.0.44, which these commits need).

Push the image to your registry and note the digest:

```bash
export WORKLOAD_IMAGE="us-docker.pkg.dev/<PROJECT>/<REPO>/maxtext-rl-ironwood:qwen3-0.6b-grpo"
docker tag maxtext_base_image ${WORKLOAD_IMAGE}
docker push ${WORKLOAD_IMAGE}
```

The launcher script in the manifest checks the installed commits and the presence
of the merged performance features at start-up (`IMAGE-CHECK` lines in the head pod
log) and exits with `FATAL: image check failed` if the pins were not moved.

## Convert the checkpoint

The run loads the official `Qwen/Qwen3-0.6B` weights converted to MaxText's scanned
layout without OCDBT or Zarr3 (the layout Pathways reads directly). Run the
conversion once, on a CPU machine with the MaxText repository checked out as above:

```bash
export CKPT_BASE="gs://<BUCKET>/checkpoints/qwen3-0.6b"
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=qwen3-0.6b \
    base_output_directory=${CKPT_BASE} \
    scan_layers=true \
    hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    --lazy_load_tensors=False \
    --eager_load_method='safetensors'

export CHECKPOINT_PATH="${CKPT_BASE}/0/items"
gcloud storage ls ${CHECKPOINT_PATH}   # must list the checkpoint files
```

The conversion stores the weights verbatim (no numeric transform), so any
MaxText-format Qwen3-0.6B checkpoint with the same layout gives the same run.

## Run the recipe

This recipe uses a Kubernetes manifest (`k8s_manifest.yaml`) to deploy the
workload. The following commands set the required environment variables,
substitute them into `k8s_manifest.yaml`, and apply the resulting configuration
to your cluster.

### 1. Configure environment variables

Open a terminal and set the following environment variables to match your setup.

**Note:**

-   `k8s_manifest.yaml` is in the same directory as this README.
-   Use a **new** `WORKLOAD_NAME` for every submission. A name that already has
    output under `BASE_OUTPUT_DIR` resumes that run's final checkpoint, trains
    nothing, and exits successfully.
-   `WORKLOAD_NAME` must be a DNS label: lowercase letters, digits and hyphens,
    at most 40 characters.

```bash
# Set variables for your environment
export PROJECT_ID=""       # Your GCP project name
export CLUSTER_NAME=""     # The name of your GKE cluster
export ZONE=""             # The zone of your GKE cluster
export BASE_OUTPUT_DIR=""  # e.g., "gs://your-bucket-name/rl-runs"
export WORKLOAD_IMAGE=""   # The image you built, e.g. "us-docker.pkg.dev/<PROJECT>/<REPO>/maxtext-rl-ironwood:qwen3-0.6b-grpo"
export CHECKPOINT_PATH=""  # The converted checkpoint, e.g. "gs://your-bucket-name/checkpoints/qwen3-0.6b/0/items"
export CPU_NODEPOOL=""     # The CPU node pool for the Pathways head pod, e.g. "cpu-pool"

# Set workload name (or modify as needed, make sure it is unique in the cluster)
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-qwen3-0.6b-grpo")-$(date +%Y%m%d-%H%M)"
```

### 2. Run the GRPO workload

Once the environment variables are set, run the following commands to fetch
cluster credentials and deploy the JobSet. Substitute only the five recipe
variables: the launcher script inside the manifest uses other `${...}` shell
variables that must stay as they are.

```bash
# Fetch cluster credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}

# Apply the manifest
envsubst '${WORKLOAD_NAME} ${WORKLOAD_IMAGE} ${BASE_OUTPUT_DIR} ${CHECKPOINT_PATH} ${CPU_NODEPOOL}' \
  < k8s_manifest.yaml | kubectl apply -n default -f -
```

If your cluster schedules TPU jobs through [Kueue](https://kueue.sigs.k8s.io/),
add the `kueue.x-k8s.io/queue-name` label to the JobSet and to both replicated job
templates before applying.

## Monitor the job

To monitor your job's progress, you can use kubectl to check the JobSet status
and logs:

```bash
# Check JobSet status
kubectl get jobset -n default ${WORKLOAD_NAME}

# The head pod runs the RL loop; follow its log
HEAD_POD=$(kubectl get pods -n default -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME},jobset.sigs.k8s.io/replicatedjob-name=pathways-head -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default ${HEAD_POD} -c jax-tpu
```

Expected timeline on a healthy cluster: the 16 worker pods and the head pod
schedule within a few minutes; the head log then shows the `IMAGE-CHECK` lines
(all `OK`), dataset and checkpoint loading, and step 1 finishing about 3 minutes
after training starts (it includes compilation). The 20-step run completes in about
25 minutes after that, and the head log ends with `MaxText Job Ended at: ... with
exit code 0` and `EXIT_CODE=0`.

You can also monitor your cluster and TPU usage through the Google Cloud
Console:
`https://console.cloud.google.com/kubernetes/workload/overview?project={PROJECT_ID}`

## Read the step time

Outputs land under `${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}/`. The step time is the
median of `perf/train/global_step_time` over TensorBoard steps 3..20; steps 1 and 2
are dropped (compilation and warm-up). `read_step_time.py` in this directory applies
exactly that rule and needs only the Python standard library:

```bash
mkdir -p tb-${WORKLOAD_NAME}
gcloud storage cp "${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}/tensorboard/events.out.tfevents.*" tb-${WORKLOAD_NAME}/
python3 read_step_time.py tb-${WORKLOAD_NAME}
# prints one line, e.g.:  perf/train/global_step_time   median 56.94  p10 55.20  p90 59.30  n=18
```

`n` must be 18. A smaller `n` means the run did not finish 20 steps; do not report
that number. On a healthy, idle slice the median should fall within about 2 s of
the reference. The `p10` / `p90` values are the spread of the individual steps
inside one run, not a tolerance for the median.

### If the step time is much slower

Every failure mode below is silent: the run starts, trains and finishes with exit
code 0, only slower. The launcher script and the manifest already set each item;
check them only if you edited the manifest.

| Symptom | Cause |
|---|---|
| Well above 80 s per step; `IMAGE-CHECK` log lines show `MISSING` | the image was built without moving the three dependency pins |
| About 35 s slower; `engine_generate` in the `PHASETIME` log lines well above 50 s | `DP_SCHED_ROUTING=round_robin` not set in the launcher |
| Very slow generation, thousands of tiny decode windows in the log | `CONTINUE_DECODE_EOS_CHECK_INTERVAL` not `"0"` |
| Train phase about 15 s instead of about 6 s | `max_seq_token_per_tpu=12288` missing (sequence packing off) |
| `compute_rewards` about 30 s instead of about 2 s | `reward_num_workers=32` missing |
| Every step 2 to 3 s slower | `free_kv_cache_during_weight_sync=false` missing (KV cache rebuilt at every weight sync) |
| First steps fine, later steps slower and slower | another job shares the TPU hosts or the head node |

## Delete resources

The JobSet and its pods are garbage-collected 24 hours after the run finishes
(`ttlSecondsAfterFinished` in the manifest). Save the head pod log before then if
you want to keep it. To delete earlier:

```bash
kubectl delete jobset -n default ${WORKLOAD_NAME}
kubectl delete configmap -n default ${WORKLOAD_NAME}-launcher
```

The TensorBoard files, checkpoints and compilation caches under
`${BASE_OUTPUT_DIR}` are not deleted by these commands.

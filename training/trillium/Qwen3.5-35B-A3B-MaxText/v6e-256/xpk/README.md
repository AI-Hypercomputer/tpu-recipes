# Pretrain qwen3.5-35b-a3b workload on Trillium GKE clusters with XPK

This recipe outlines the end-to-end steps for running a **Qwen3.5-35B-A3B**
(`qwen3.5-35b-a3b`) [MaxText](https://github.com/AI-Hypercomputer/maxtext)
pretraining workload on [Trillium GKE clusters](https://cloud.google.com/kubernetes-engine)
(`v6e-256`) using [XPK](https://github.com/AI-Hypercomputer/xpk).

## Workload Details

This workload is configured with the following details:

-   Model: `qwen3.5-35b-a3b`
-   Sequence Length: `8192`
-   Per-Device Batch Size: `8` (`global_batch_size=2048`)
-   Precision: `bfloat16`
-   Chips: `256` (`v6e-256`, `16x16` topology across `64` `ct6e-standard-4t` hosts)
-   Dataset: Synthetic mock dataset provided by MaxText (`dataset_type=synthetic`)

## Prerequisites

Ensure the following are set up before starting:

-   **GCP Project Setup:** A GCP project with billing enabled and quota or a
    reservation for Cloud TPU Trillium (`v6e-256`).
-   **Workstation Tools:**
    -   [`gcloud`](https://cloud.google.com/sdk/docs/install), `kubectl`, and
        `gke-gcloud-auth-plugin` installed and authenticated (`gcloud auth login`
        and `gcloud auth application-default login`).
    -   Docker installed and running on your workstation.
    -   [`uv`](https://docs.astral.sh/uv/) with Python `3.12` (for building the
        MaxText container image).
    -   [`xpk`](https://github.com/AI-Hypercomputer/xpk) (`v1.11.0`, along with
        `kubectl-kueue` and `kjob`) installed and available in your `PATH`.
-   **Verified Software Stack:**
    -   **MaxText:** `main` branch (`https://github.com/AI-Hypercomputer/maxtext`)
    -   **JAX / Libtpu:** MaxText nightly stack (`MODE=nightly`, validated on `jax 0.12.0.dev20261001`)
    -   **Container Python:** `3.12`
    -   **Host XPK CLI:** `1.11.0`

## Orchestration and deployment tools

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Pretraining job configuration and deployment** - XPK configures and
    deploys a
    [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource to execute the `qwen3.5-35b-a3b` MaxText pretraining workload across
    64 TPU hosts (`256` chips).

## 1. Set Environment Variables

Set the following environment variables in your shell and keep `PROJECT_ID`,
`CLUSTER_NAME`, `ZONE`, `BASE_OUTPUT_DIR`, and `WORKLOAD_IMAGE` consistent
across cluster creation, Docker image build, and workload submission:

-   `PROJECT_ID`: Your GCP project ID.
-   `CLUSTER_NAME`: Target GKE cluster name.
-   `ZONE`: Compute zone containing your `v6e-256` capacity (e.g., `us-east5-b`).
-   `REGION`: Artifact Registry region (derived as `${ZONE%-*}` by default, e.g., `us-east5`).
-   `REPO_NAME`: Artifact Registry Docker repository name (e.g., `maxtext-runner`).
-   `BASE_OUTPUT_DIR`: Cloud Storage output directory (e.g., `gs://<YOUR_BUCKET_NAME>/qwen35-35b-a3b-output`).
-   `WORKLOAD_IMAGE`: Full container image URI in Artifact Registry:
    `${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/qwen35-35b-a3b-runner:latest`.
-   `WORKLOAD_NAME`: Unique JobSet workload name (defaults to
    `qwen35-35b-v6e256-$(date +%m%d-%H%M)` in `run_recipe.sh`).
-   `ACCELERATOR_TYPE`: `v6e-256` (see [TPU topologies](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#configuration)).
-   `RESERVATION_NAME`: Your TPU reservation name (or `projects/<project_number>/reservations/<reservation_name>` for shared reservations).

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export ZONE="<YOUR_ZONE>" # e.g., us-east5-b
export REGION="${ZONE%-*}" # e.g., us-east5
export REPO_NAME="maxtext-runner"
export BASE_OUTPUT_DIR="gs://<YOUR_BUCKET_NAME>/qwen35-35b-a3b-output"
export WORKLOAD_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/qwen35-35b-a3b-runner:latest"
export ACCELERATOR_TYPE="v6e-256"
export RESERVATION_NAME="<YOUR_RESERVATION_NAME>"
```

If you need to create a Cloud Storage bucket for `BASE_OUTPUT_DIR`:

```bash
gcloud storage buckets create gs://<YOUR_BUCKET_NAME> \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access
```

## 2. Create the GKE Cluster

If you do not already have a `v6e-256` GKE cluster, create one with XPK (omit
`--reservation=${RESERVATION_NAME}` if using on-demand TPU capacity):

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=${ACCELERATOR_TYPE} \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

## 3. Build and Push the Docker Container Image

Build the MaxText nightly base image and upload the workload runner image to
Artifact Registry:

```bash
# (Optional) Create the Artifact Registry Docker repository if needed
gcloud artifacts repositories create ${REPO_NAME} \
  --project=${PROJECT_ID} \
  --repository-format=docker \
  --location=${REGION}
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Create and activate a Python 3.12 virtual environment for the image build
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.12 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

# Clone MaxText and upload the workload runner image
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=nightly
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${WORKLOAD_IMAGE}

deactivate
cd ..
```

## 4. Get and Run the Recipe

Clone `tpu-recipes` and navigate to this recipe directory:

```bash
git clone https://github.com/AI-Hypercomputer/tpu-recipes.git
cd tpu-recipes/training/trillium/Qwen3.5-35B-A3B-MaxText/v6e-256/xpk
```

(Optional) Fetch `kubectl` cluster credentials manually if needed (XPK also configures credentials automatically):

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE}
```

Ensure your environment variables from [1. Set Environment Variables](#1-set-environment-variables)
are exported in your shell (or set directly at the top of `run_recipe.sh`), then
launch the workload:

```bash
chmod +x run_recipe.sh
./run_recipe.sh
```

You can customize the benchmark inside `run_recipe.sh`:

-   **Environment Variables:** `PROJECT_ID`, `CLUSTER_NAME`, `ZONE`,
    `BASE_OUTPUT_DIR`, `WORKLOAD_IMAGE`, and `WORKLOAD_NAME` inherit values
    already exported in your shell or can be edited directly at the top of
    `run_recipe.sh`.
-   **XLA Flags:** `XLA_FLAGS` configures TPU v6e VMEM limits, async collective
    pipelining, and SparseCore all-reduce offload.
-   **MaxText Overrides:** `MAXTEXT_ARGS` sets `qwen3.5-35b-a3b` sharding,
    sub-layer rematerialization, fused Gated DeltaNet Pallas kernels, MoE Pallas
    gather/combine kernels, and Megablox tile sizes.

## 5. Monitor the Job

Check JobSet status and stream logs from worker 0:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} \
  -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default ${POD_NAME}
```

You can also list or inspect workloads directly via XPK:

```bash
xpk workload list --cluster=${CLUSTER_NAME} --project=${PROJECT_ID} --zone=${ZONE}
xpk inspector --cluster=${CLUSTER_NAME} --project=${PROJECT_ID} --zone=${ZONE} --workload=${WORKLOAD_NAME}
```

## 6. Check Results

After training steps begin, review your steady-state step times (`steps 15–29`)
and `TFLOP/s/device` in the pod logs:

```text
completed step: 29, seconds: 9.248, TFLOP/s/device: 138.643, Tokens/s/device: 7086.440, total_weights: 16777216, loss: 12.699
```

## 7. Delete Resources

### Delete a specific workload

```bash
xpk workload delete \
  --workload=${WORKLOAD_NAME} \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE}
```

### Delete the XPK cluster

```bash
xpk cluster delete \
  --cluster=${CLUSTER_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID}
```

## Next steps: deeper exploration and customization

For custom experiments, dataset integration, and checkpointing configurations,
see the
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md).

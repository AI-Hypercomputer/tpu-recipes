# Pretrain qwen3.5-35b-a3b workload on Trillium GKE clusters with Kubernetes JobSet

This recipe outlines the end-to-end steps for running a **Qwen3.5-35B-A3B**
(`qwen3.5-35b-a3b`) [MaxText](https://github.com/AI-Hypercomputer/maxtext)
pretraining workload on [Trillium GKE clusters](https://cloud.google.com/kubernetes-engine)
(`v6e-256`) by provisioning a GKE TPU slice, building the MaxText container
image, and applying `k8s_manifest.yaml` to deploy a Kubernetes JobSet resource.

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
    -   `envsubst` installed (`gettext-base` on Debian/Ubuntu).
-   **Verified Software Stack:**
    -   **MaxText:** `main` branch (`https://github.com/AI-Hypercomputer/maxtext`)
    -   **JAX / Libtpu:** MaxText nightly stack (`MODE=nightly`, validated on `jax 0.12.0.dev20261001`)
    -   **Container Python:** `3.12`
    -   **Kubernetes JobSet:** `v0.13.0`

## Orchestration and deployment tools

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Pretraining job configuration and deployment** - A Kubernetes manifest
    (`k8s_manifest.yaml`) defines and deploys the
    [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource managing all 64 worker pods (`256` TPU v6e chips).

## 1. Set Environment Variables

Set the following environment variables in your shell and keep them consistent
across cluster creation, Docker image build, and JobSet submission:

-   `PROJECT_ID`: Your GCP project ID.
-   `CLUSTER_NAME`: Target GKE cluster name.
-   `ZONE`: Compute zone containing your `v6e-256` capacity (e.g., `us-east5-b`).
-   `REGION`: Artifact Registry region (derived as `${ZONE%-*}` by default, e.g., `us-east5`).
-   `NODE_POOL_NAME`: Name of the `v6e-256` TPU node pool (e.g., `tpu-v6e-256-pool`).
-   `REPO_NAME`: Artifact Registry Docker repository name (e.g., `maxtext-runner`).
-   `BASE_OUTPUT_DIR`: Cloud Storage output directory (e.g., `gs://<YOUR_BUCKET_NAME>/qwen35-35b-a3b-output`).
-   `WORKLOAD_IMAGE`: Full container image URI in Artifact Registry:
    `${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/qwen35-35b-a3b-runner:latest`.
-   `WORKLOAD_NAME`: Unique JobSet name in the cluster.
-   `RESERVATION_NAME`: Your TPU reservation name (or `projects/<project_number>/reservations/<reservation_name>` for shared reservations).

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export ZONE="<YOUR_ZONE>" # e.g., us-east5-b
export REGION="${ZONE%-*}" # e.g., us-east5
export NODE_POOL_NAME="tpu-v6e-256-pool"
export REPO_NAME="maxtext-runner"
export BASE_OUTPUT_DIR="gs://<YOUR_BUCKET_NAME>/qwen35-35b-a3b-output"
export WORKLOAD_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/qwen35-35b-a3b-runner:latest"
export WORKLOAD_NAME="qwen35-35b-v6e256-$(date +%m%d-%H%M)"
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

## 2. Create the GKE Cluster, `v6e-256` Node Pool, and Install JobSet

If you do not already have a GKE cluster with a `v6e-256` (`16x16`) node pool
and the JobSet controller installed, run the commands below.

1. Create a Standard GKE cluster:

```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --release-channel=rapid \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

2. Create the `v6e-256` TPU slice node pool (`64` `ct6e-standard-4t` VMs in a
   `16x16` topology; omit `--reservation-affinity=specific` and
   `--reservation=${RESERVATION_NAME}` if using on-demand TPU capacity):

```bash
gcloud container node-pools create ${NODE_POOL_NAME} \
  --project=${PROJECT_ID} \
  --cluster=${CLUSTER_NAME} \
  --zone=${ZONE} \
  --machine-type=ct6e-standard-4t \
  --tpu-topology=16x16 \
  --num-nodes=64 \
  --reservation-affinity=specific \
  --reservation=${RESERVATION_NAME} \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

3. Configure `kubectl` credentials and install the Kubernetes
   [JobSet](https://jobset.sigs.k8s.io/docs/installation/) controller if it is
   not already installed on your cluster:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID}

export JOBSET_VERSION="v0.13.0"
kubectl apply --server-side \
  -f https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml

kubectl wait --for=condition=Available deployment/jobset-controller-manager \
  -n jobset-system --timeout=120s
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
cd tpu-recipes/training/trillium/Qwen3.5-35B-A3B-MaxText/v6e-256/k8s
```

Ensure your environment variables from [1. Set Environment Variables](#1-set-environment-variables)
are exported in your shell, configure `kubectl` credentials if needed, and
apply `k8s_manifest.yaml`:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID}

envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE}' \
  < k8s_manifest.yaml | kubectl apply -n default -f -
```

You can customize the benchmark inside `k8s_manifest.yaml`:

-   **Template Variables:** `${WORKLOAD_NAME}`, `${WORKLOAD_IMAGE}`, and
    `${BASE_OUTPUT_DIR}` are substituted from your shell environment via
    `envsubst`.
-   **XLA Flags:** `LIBTPU_INIT_ARGS` configures TPU v6e VMEM limits, async
    collective pipelining, and SparseCore all-reduce offload.
-   **MaxText Overrides:** The arguments passed to
    `python3 -u -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml`
    configure `qwen3.5-35b-a3b` sharding, sub-layer rematerialization, fused
    Gated DeltaNet Pallas kernels, MoE Pallas gather/combine kernels, and
    Megablox tile sizes.

## 5. Monitor the Job

Check the JobSet status and follow worker 0 logs with `kubectl`:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} \
  -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default ${POD_NAME}
```

You can also view workload status in the Google Cloud Console:
`https://console.cloud.google.com/kubernetes/workload/overview?project=${PROJECT_ID}`

## 6. Check Results

Once training steps begin, steady-state logs (`steps 15–29`) will report step
duration and throughput:

```text
completed step: 29, seconds: 9.248, TFLOP/s/device: 138.643, Tokens/s/device: 7086.440, total_weights: 16777216, loss: 12.699
```

## 7. Delete Resources

### Delete a specific workload

To delete the JobSet deployed by this recipe:

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default
```

### Delete the GKE TPU node pool or cluster

To avoid recurring charges when finished, delete the `v6e-256` node pool or the
entire GKE cluster:

```bash
gcloud container node-pools delete ${NODE_POOL_NAME} \
  --cluster=${CLUSTER_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID}

gcloud container clusters delete ${CLUSTER_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID}
```

## Next steps: deeper exploration and customization

For custom experiments, dataset integration, and checkpointing configurations,
see the
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md).

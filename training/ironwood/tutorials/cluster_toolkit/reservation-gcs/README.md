# Run a training workload on Ironwood (TPU v7x) with a reservation and Cloud Storage using Cluster Toolkit

This tutorial walks through an end-to-end flow on Ironwood (Cloud TPU v7x) GKE
clusters by using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`) and the `examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml` blueprint:

1.  Deploy a GKE cluster with a TPU 7x node pool bound to a **reservation**
    (regular, gSC, or DWS Calendar), multi-NIC networking (DRANET), Kueue, and
    JobSet.
2.  Provision hierarchical-namespace **Cloud Storage buckets** and
    performance-tuned GCS FUSE Storage Profile PersistentVolumeClaims
    (`gcsfusecsi-training` and `gcsfusecsi-checkpointing`) automatically via the
    blueprint.
3.  Run a mock training workload, then a small
    [MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining
    workload, with `gcluster job submit`.

For a full model recipe that uses Cloud Storage on Ironwood, see
[DeepSeek-V3 671B 4x8x8 (GCS) with Cluster Toolkit](../../../deepseek3-671b/4k-bf16-tpu7x-4x8x8-gcs/cluster_toolkit/README.md).

## Before you begin

-   A Google Cloud project with billing enabled and access to TPU 7x. Contact
    your account team for Ironwood access.
-   A TPU 7x reservation in the project (or shared with it). The reservation can
    be a regular reservation, a gSC reservation, or a DWS Calendar reservation.
-   The account you use needs the following IAM roles:
    -   Compute Admin
    -   Kubernetes Engine Admin
    -   Project IAM Admin (or Security Admin, to grant roles to the blueprint's
        node pool and workload service accounts)
    -   Service Account Admin and Service Account User
    -   Storage Admin
    -   Artifact Registry Writer (only if you build the MaxText image)
    -   Logging Admin and Monitoring Admin
-   `gcloud`, `kubectl`, `gke-gcloud-auth-plugin`, and `terraform` installed and
    authenticated. Follow the
    [Cluster Toolkit environment setup guide](https://docs.cloud.google.com/cluster-toolkit/docs/setup/configure-environment).
-   Docker and Python 3.12 (`python3.12` with `venv`) if you build the MaxText
    container image in Option B.

### Install Cluster Toolkit (gcluster)

```bash
# Set Cluster Toolkit version
export CTK_VERSION="1.108.0"

# Download the prebuilt bundle from GitHub releases
curl -L -O "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/v${CTK_VERSION}/gcluster_bundle_linux_amd64.tgz"

# Extract the bundle
mkdir -p "${HOME}/cluster-toolkit"
tar -xzf gcluster_bundle_linux_amd64.tgz -C "${HOME}/cluster-toolkit"
rm gcluster_bundle_linux_amd64.tgz

# Add gcluster to your PATH
export PATH="${HOME}/cluster-toolkit:${PATH}"
echo 'export PATH="${HOME}/cluster-toolkit:${PATH}"' >> ~/.bashrc

# Verify installation
gcluster --version
```

The bundle ships the blueprints under `${HOME}/cluster-toolkit/examples/`,
which the commands below reference.

## Set environment variables

> **NOTE:** For multi-host provisioning use a topology that results in more
> than 4 chips (for example `2x2x2` = 8 chips across 2 `tpu7x-standard-4t`
> nodes). For single-host provisioning use `2x2x1` (4 chips on 1 node). See the
> [TPU 7x configurations](https://docs.cloud.google.com/tpu/docs/tpu7x#configurations).

```bash
export PROJECT_ID=<project_id>            # Your Google Cloud project ID
export ZONE=<zone>                        # Example: us-central1-c
export REGION="${ZONE%-*}"
export CLUSTER_NAME=<cluster_name>        # Unique Cluster Toolkit deployment name
export TPU_TOPOLOGY=<topology>            # Example: 2x2x2
export RESERVATION_NAME=<reservation>     # Bare reservation name (no "projects/..." path)
export AUTHORIZED_CIDR="$(curl -s https://ifconfig.me)/32"  # Caller IP for kubectl access (or 0.0.0.0/0 for testing)
export TF_STATE_BUCKET="${PROJECT_ID}-ctk-tf-state"
```

## Create the cluster and Cloud Storage volumes

### 1. Enable APIs, create the Terraform state bucket, and authenticate

```bash
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  iam.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

gcloud storage buckets create "gs://${TF_STATE_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access

gcloud storage buckets update "gs://${TF_STATE_BUCKET}" --versioning

# Application Default Credentials for Terraform
gcloud auth application-default login
```

### 2. Write the deployment configuration and deploy `gke-tpu-7x-advanced.yaml`

The `gke-tpu-7x-advanced.yaml` blueprint provisions:

-   The VPC network, GKE cluster (`REGULAR` channel, GKE 1.35+, maintenance
    exclusions, Dataplane V2, and Workload Identity), and a TPU 7x node pool
    with DRANET (`enable_dranet: true`) bound to your reservation via
    `SPECIFIC_RESERVATION` affinity. Cluster Toolkit derives the node count
    automatically from `tpu_topology` and `compute_machine_type`.
-   Kueue (`user-queue`) and JobSet.
-   Two hierarchical-namespace Cloud Storage buckets (`training-data-<suffix>`
    and `checkpoint-data-<suffix>`) along with bound GCS FUSE Storage Profile
    PVCs (`gcsfusecsi-training` and `gcsfusecsi-checkpointing`) and a Workload
    Identity service account (`workload-identity-k8s-sa`) granted
    `roles/storage.admin`.

Create a deployment file (matching the schema of
`examples/gke-tpu-7x/gke-tpu-7x-deployment.yaml`) and deploy the advanced
blueprint:

```bash
cd "${HOME}/cluster-toolkit"

cat <<EOF > gke-tpu-7x-reservation-gcs-deployment.yaml
terraform_backend_defaults:
  type: gcs
  configuration:
    bucket: ${TF_STATE_BUCKET}

vars:
  project_id: ${PROJECT_ID}
  deployment_name: ${CLUSTER_NAME}
  region: ${REGION}
  zone: ${ZONE}
  num_slices: 1
  compute_machine_type: tpu7x-standard-4t
  tpu_topology: ${TPU_TOPOLOGY}
  authorized_cidr: ${AUTHORIZED_CIDR}
  user_namespace: default
  reservation_affinity:
    consume_reservation_type: SPECIFIC_RESERVATION
    specific_reservations:
    - name: ${RESERVATION_NAME}
EOF
```

> **Shared reservations:** If the reservation lives in a different host project,
> add `project: <reservation_host_project_id>` directly beneath
> `- name: ${RESERVATION_NAME}` in `specific_reservations` rather than embedding
> a `projects/.../reservations/...` path in `name`.
>
> **System node pool machine type:** If your zone does not offer the default
> `n2d-standard-8` CPU machine type, add
> `system_node_pool_machine_type: n2-standard-8` (or `e2-standard-8`) under
> `vars:`.

Deploy the blueprint:

```bash
./gcluster deploy \
  -d gke-tpu-7x-reservation-gcs-deployment.yaml \
  examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml
```

When prompted, select **(A)pply**. For the full list of blueprint options, see
the
[Cloud TPU 7x (Ironwood) GKE deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x).

### 3. Connect to the cluster and inspect provisioned PVCs

```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# TPU nodes should be Ready
kubectl get nodes

# Retrieve the Kueue LocalQueue name (defaults to "user-queue")
kubectl get localqueues -n default
export QUEUE=$(kubectl get localqueues -n default -o jsonpath='{.items[0].metadata.name}')

# Retrieve the checkpoint PVC and Cloud Storage bucket created by the blueprint
kubectl get pvc -n default
export CHECKPOINT_PVC=$(kubectl get pvc -n default \
  -o jsonpath='{.items[?(@.spec.storageClassName=="gcsfusecsi-checkpointing")].metadata.name}')
export OUTPUT_BUCKET="${CHECKPOINT_PVC%-pvc}"
export MOUNT_PATH="/checkpoint-data"
echo "Checkpoint PVC: ${CHECKPOINT_PVC} -> gs://${OUTPUT_BUCKET}"
```

> **Using an existing bucket instead:** You can also mount any existing Cloud
> Storage bucket directly in `gcluster job submit` without a pre-created PVC by
> passing `--mount "gs://<bucket_name>;/checkpoint-data;rw;profile=checkpointing"`
> (as long as `workload-identity-k8s-sa`'s backing GCP service account has access
> to that bucket).

## Run a workload

Pass `--service-account workload-identity-k8s-sa` so pods authenticate via
Workload Identity as the blueprint's `gke-wl-sa` service account (the node
pool's `gke-np-sa` service account only has read-only `storage.objectViewer`
permissions).

<details>
<summary><strong>Option A: Mock training workload</strong></summary>

This runs a CPU-only script on the TPU node pool to validate scheduling,
reservation affinity, Workload Identity, and the GCS FUSE mount. Each container
writes a marker file to the checkpoint bucket.

```bash
export WORKLOAD_NAME="tf-mock-$(date +%H%M)"

gcluster job submit \
  --skip-prereqs \
  --queue "${QUEUE}" \
  --cluster "${CLUSTER_NAME}" \
  --project "${PROJECT_ID}" \
  --location "${ZONE}" \
  --compute-type tpu7x \
  --topology "${TPU_TOPOLOGY}" \
  --num-slices 1 \
  --restarts 0 \
  --service-account workload-identity-k8s-sa \
  --image python:3.12-slim \
  --mount "${CHECKPOINT_PVC};${MOUNT_PATH};rw" \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "python3 -u -c \"import math, time; [time.sleep(0.5) or print(f'Epoch {e}/5 - loss: {2.15 * math.exp(-0.35 * e) + 0.15:.4f}') for e in range(1, 6)]\" && \
echo done > ${MOUNT_PATH}/${WORKLOAD_NAME}-\${TPU_WORKER_ID:-\${JOB_COMPLETION_INDEX:-0}}-\$\$.txt && \
ls -l ${MOUNT_PATH}"
```

</details>

<details>
<summary><strong>Option B: Train a small model with MaxText</strong></summary>

1.  Build and push the MaxText image. MaxText requires **Python 3.12**.

    ```bash
    export CONTAINER_REGISTRY=<registry>   # Example: us-docker.pkg.dev/<project>/<repo> or gcr.io
    export CLOUD_IMAGE_NAME="${USER}-maxtext-runner"
    export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

    # Set up and activate a Python 3.12 virtual environment for the Docker build
    python3.12 -m venv "${HOME}/.local/bin/venv-docker"
    source "${HOME}/.local/bin/venv-docker/bin/activate"
    pip install --upgrade pip

    git clone https://github.com/AI-Hypercomputer/maxtext.git
    cd maxtext
    git checkout cf051eb03

    bash src/dependencies/scripts/docker_build_dependency_image.sh \
      MODE=stable \
      JAX_VERSION=0.8.1 \
      LIBTPU_VERSION=0.0.30
    bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

    deactivate
    cd ..
    ```

1.  Submit a MaxText pretraining run on a synthetic dataset. Outputs are written
    to the Cloud Storage bucket through the GCS FUSE mount.

    ```bash
    export WORKLOAD_NAME="maxtext-$(date +%H%M)"
    export BASE_OUTPUT_DIR="${MOUNT_PATH}"

    export MAXTEXT_ARGS="\
      base_output_directory=${BASE_OUTPUT_DIR} \
      dataset_type=synthetic \
      per_device_batch_size=2 \
      enable_checkpointing=false \
      run_name=${WORKLOAD_NAME} \
      steps=30"

    gcluster job submit \
      --skip-prereqs \
      --queue "${QUEUE}" \
      --cluster "${CLUSTER_NAME}" \
      --project "${PROJECT_ID}" \
      --location "${ZONE}" \
      --priority medium \
      --restarts 0 \
      --compute-type tpu7x \
      --topology "${TPU_TOPOLOGY}" \
      --num-slices 1 \
      --service-account workload-identity-k8s-sa \
      --image "${WORKLOAD_IMAGE}" \
      --mount "${CHECKPOINT_PVC};${MOUNT_PATH};rw" \
      --verbose \
      --gke-namespace default \
      --name "${WORKLOAD_NAME}" \
      --command "set -e && export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
    python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS}"
    ```

</details>

<details>
<summary><strong>Option C: Train Llama 3.1 70B with MaxText</strong></summary>

Llama 3.1 70B needs at least a `4x4x4` topology (64 chips). Recreate the cluster
with a larger `TPU_TOPOLOGY` if needed, then follow the
[Llama 3.1 70B 4x4x4 Cluster Toolkit recipe](../../../llama3.1-70b/8k-bf16-tpu7x-4x4x4/cluster_toolkit/README.md),
setting `BASE_OUTPUT_DIR` in its `run_recipe.sh` to `gs://${OUTPUT_BUCKET}` (and
adding `--service-account workload-identity-k8s-sa` and `--queue "${QUEUE}"` to
`gcluster job submit`).

</details>

## Monitor the workload

```bash
gcluster job list --cluster "${CLUSTER_NAME}" --project "${PROJECT_ID}" --location "${ZONE}"
gcluster job logs "${WORKLOAD_NAME}" --cluster "${CLUSTER_NAME}" --project "${PROJECT_ID}" --location "${ZONE}"

# Or with kubectl
kubectl get jobset -n default "${WORKLOAD_NAME}"
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${WORKLOAD_NAME}" -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default "${POD_NAME}"

# Verify outputs in the Cloud Storage bucket
gcloud storage ls "gs://${OUTPUT_BUCKET}/"
```

## Clean up

Because `gke-tpu-7x-advanced.yaml` sets `force_destroy: false` on its
`training_bucket` and `checkpoint_bucket` modules to protect data against
accidental deletion, empty any objects written to `gs://${OUTPUT_BUCKET}` before
running `gcluster destroy`:

```bash
# Cancel an active workload (if still running)
gcluster job cancel "${WORKLOAD_NAME}" --cluster "${CLUSTER_NAME}" --project "${PROJECT_ID}" --location "${ZONE}"

# Empty objects written during the tutorial so Terraform can delete the bucket
gcloud storage rm -r "gs://${OUTPUT_BUCKET}/**" 2>/dev/null || true

# Destroy the cluster, PVCs/PVs, buckets, and networking created by the blueprint
cd "${HOME}/cluster-toolkit"
./gcluster destroy "${CLUSTER_NAME}" --auto-approve

# Optionally delete the Terraform state bucket
gcloud storage rm -r "gs://${TF_STATE_BUCKET}"
```

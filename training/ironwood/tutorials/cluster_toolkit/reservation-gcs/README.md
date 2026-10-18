# Run a training workload on Ironwood (TPU v7x) with a reservation and Cloud Storage using Cluster Toolkit

This tutorial walks through an end-to-end flow on Ironwood (Cloud TPU v7x) GKE
clusters by using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`):

1.  Create a GKE cluster with a TPU 7x node pool backed by a **reservation**
    (regular, gSC, or DWS Calendar).
2.  Use a **Cloud Storage bucket** as the workload's output and checkpoint
    location, mounted into the container with the GCS FUSE CSI driver.
3.  Run a mock training workload, then a small
    [MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining
    workload, with `gcluster job submit`.

It is the Cluster Toolkit equivalent of the XPK
[reservation + GCS bucket recipe](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/tpu7x/recipes/reservation_gcs_bucket_recipe.md).

For a full model recipe that uses this setup, see
[DeepSeek-V3 671B 4x8x8 (GCS) with Cluster Toolkit](../../../deepseek3-671b/4k-bf16-tpu7x-4x8x8-gcs/cluster_toolkit/README.md).

## Before you begin

-   A Google Cloud project with billing enabled and access to TPU 7x. Contact
    your account team for Ironwood access.
-   A TPU 7x reservation in the project (or shared with it). The reservation can
    be a regular reservation, a gSC reservation, or a DWS Calendar reservation.
-   The account you use needs the following IAM roles:
    -   Compute Admin
    -   Kubernetes Engine Admin
    -   Service Account Admin and Service Account User
    -   Storage Admin
    -   Artifact Registry Writer (only if you build the MaxText image)
    -   Logging Admin and Monitoring Admin
-   `gcloud`, `kubectl`, and `gke-gcloud-auth-plugin` installed and
    authenticated. Follow the
    [Cluster Toolkit environment setup guide](https://docs.cloud.google.com/cluster-toolkit/docs/setup/configure-environment).

### Install Cluster Toolkit (gcluster)

```bash
# Set Cluster Toolkit version
export CTK_VERSION="1.105.0"

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
> than 8 chips (for example `2x2x2`). For single-host provisioning use a
> topology with 8 or fewer chips (for example `2x2x1`). See the
> [TPU 7x configurations](https://docs.cloud.google.com/tpu/docs/tpu7x#configurations).

```bash
export PROJECT_ID=<project_id>            # Your Google Cloud project ID
export ZONE=<zone>                        # Example: us-central1-c
export REGION="${ZONE%-*}"
export CLUSTER_NAME=<cluster_name>        # Also used as the Cluster Toolkit deployment name
export TPU_TOPOLOGY=<topology>            # Example: 2x2x2
export RESERVATION_NAME=<reservation>     # Reservation name. For a shared reservation use
                                          # "projects/<project_number>/reservations/<reservation_name>"
export OUTPUT_BUCKET=<bucket_name>        # Cloud Storage bucket name, without the gs:// prefix
export TF_STATE_BUCKET="${PROJECT_ID}-ctk-tf-state"
```

## Create the cluster with a reservation

### 1. Create the Terraform state bucket and credentials

```bash
gcloud storage buckets create "gs://${TF_STATE_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access

gcloud storage buckets update "gs://${TF_STATE_BUCKET}" --versioning

# Application Default Credentials for Terraform
gcloud auth application-default login
```

### 2. Deploy the TPU 7x blueprint

The `gke-tpu-7x` blueprint provisions the VPCs, the GKE cluster, a TPU 7x node
pool bound to your reservation with `SPECIFIC_RESERVATION` affinity, Kueue, and
JobSet. Cluster Toolkit derives the node count from `tpu_topology` and
`machine_type`.

```bash
cd ~/cluster-toolkit
./gcluster deploy examples/gke-tpu-7x/gke-tpu-7x.yaml \
  --backend-config="bucket=${TF_STATE_BUCKET}" \
  --vars="project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME},region=${REGION},zone=${ZONE},num_slices=1,machine_type=tpu7x-standard-4t,tpu_topology=${TPU_TOPOLOGY},reservation=${RESERVATION_NAME}"
```

When prompted, select **(A)pply**. For the full list of blueprint options, see
the
[Cloud TPU 7x (Ironwood) GKE deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x).

### 3. Connect to the cluster

```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# TPU nodes should be Ready
kubectl get nodes

# Note the Kueue LocalQueue name; it is passed to gcluster job submit below
kubectl get localqueues -n default
export QUEUE=$(kubectl get localqueues -n default -o jsonpath='{.items[0].metadata.name}')
```

## Set up Cloud Storage

Create the bucket that the workload will write outputs and checkpoints to. The
blueprint enables the GCS FUSE CSI driver on the cluster and grants the workload
service account `roles/storage.admin`, so no PersistentVolume is needed:
`gcluster job submit --mount` mounts the bucket inline.

```bash
gcloud storage buckets create "gs://${OUTPUT_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access

export MOUNT_PATH="/data-gcs"
export MOUNT_OPTIONS="implicit-dirs,metadata-cache:ttl-secs:-1,metadata-cache:stat-cache-max-size-mb:-1,metadata-cache:type-cache-max-size-mb:-1,write:enable-streaming-writes:true"
```

## Run a workload

<details>
<summary><strong>Option A: Mock training workload</strong></summary>

This runs a CPU-only script on the TPU node pool to validate that scheduling,
the reservation, and the Cloud Storage mount all work. The script writes a
marker file to the bucket.

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
  --image python:3.12-slim \
  --mount "gs://${OUTPUT_BUCKET};${MOUNT_PATH};rw;options=${MOUNT_OPTIONS}" \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "python3 -c \"import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/heads/main/examples/fake_training.py', 'fake_training.py')\" && \
python3 fake_training.py && \
echo done > ${MOUNT_PATH}/${WORKLOAD_NAME}-\${JOB_COMPLETION_INDEX:-0}.txt && \
ls -l ${MOUNT_PATH}"
```

</details>

<details>
<summary><strong>Option B: Train a small model with MaxText</strong></summary>

1.  Build and push the MaxText image. MaxText requires **Python 3.12**.

    ```bash
    export CONTAINER_REGISTRY=<registry>   # Example: gcr.io
    export CLOUD_IMAGE_NAME="${USER}-maxtext-runner"
    export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

    # Python 3.12 virtual environment for the Docker build
    uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.12 --clear
    source ${HOME}/.local/bin/venv-docker/bin/activate
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
      --image "${WORKLOAD_IMAGE}" \
      --mount "gs://${OUTPUT_BUCKET};${MOUNT_PATH};rw;options=${MOUNT_OPTIONS}" \
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
setting `BASE_OUTPUT_DIR` in its `run_recipe.sh` to `gs://${OUTPUT_BUCKET}`.

</details>

## Monitor the workload

```bash
gcluster job list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
gcluster job logs ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}

# Or with kubectl
kubectl get jobset -n default ${WORKLOAD_NAME}
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default ${POD_NAME}

# Outputs land in the bucket
gcloud storage ls "gs://${OUTPUT_BUCKET}/"
```

## Clean up

```bash
# Delete a workload
gcluster job cancel ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}

# Destroy the cluster and all resources created by the blueprint
cd ~/cluster-toolkit
./gcluster destroy ${CLUSTER_NAME} --auto-approve

# Optionally delete the buckets
gcloud storage rm -r "gs://${OUTPUT_BUCKET}"
gcloud storage rm -r "gs://${TF_STATE_BUCKET}"
```

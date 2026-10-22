# Run a training workload on Ironwood (TPU v7x) with flex-start and Filestore using Cluster Toolkit

This tutorial walks through an end-to-end flow on Ironwood (Cloud TPU v7x) GKE
clusters by using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`):

1.  Create a GKE cluster whose TPU 7x node pool is provisioned on demand with
    [Dynamic Workload Scheduler flex-start](https://cloud.google.com/kubernetes-engine/docs/how-to/dws-flex-start-training).
    The node pool starts at **0 nodes** and scales up when a workload is
    submitted.
2.  Create a **Filestore** instance and expose it to the cluster as a shared
    `ReadWriteMany` PersistentVolume.
3.  Run a mock training workload, then a small
    [MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining
    workload, with `gcluster job submit`.

It is the Cluster Toolkit equivalent of the XPK
[flex-start + Filestore recipe](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/tpu7x/recipes/flex_filestore_recipe.md).

## Before you begin

-   A Google Cloud project with billing enabled and access to TPU 7x. Contact
    your account team for Ironwood access.
-   The account you use needs the following IAM roles:
    -   Compute Admin
    -   Kubernetes Engine Admin
    -   Service Account Admin and Service Account User
    -   Storage Admin
    -   Cloud Filestore Editor
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
>
> With flex-start, the node pool autoscales between 0 and `MAX_NODES`.
> `MAX_NODES` must equal the number of nodes in your slice: total chips in
> `TPU_TOPOLOGY` divided by 4 chips per `tpu7x-standard-4t` node (for example,
> `2x2x2` = 8 chips = 2 nodes).

```bash
export PROJECT_ID=<project_id>            # Your Google Cloud project ID
export ZONE=<zone>                        # Example: us-central1-c
export REGION="${ZONE%-*}"
export CLUSTER_NAME=<cluster_name>        # Also used as the Cluster Toolkit deployment name
export TPU_TOPOLOGY=<topology>            # Example: 2x2x2
export MAX_NODES=<node_count>             # Example: 2 for a 2x2x2 topology
export NETWORK_NAME="${CLUSTER_NAME}-net-0"   # Primary VPC created by the blueprint
export STORAGE_NAME=<filestore_instance>  # Filestore instance name
export FILE_SHARE_NAME="default"
export TF_STATE_BUCKET="${PROJECT_ID}-ctk-tf-state"
```

## Create the cluster with flex-start

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

### 2. Enable the Filestore CSI driver in the blueprint

The DWS flex-start TPU 7x blueprint enables the GCS FUSE CSI driver but not the
Filestore CSI driver. Add `enable_filestore_csi: true` to the
`gke-tpu-7x-cluster` module settings in
`~/cluster-toolkit/examples/gke-consumption-options/dws-flex-start/gke-tpu-7x/gke-tpu-7x.yaml`:

```yaml
  - id: gke-tpu-7x-cluster
    source: modules/scheduler/gke-cluster
    use: [gke-tpu-7x-net-0, workload_service_account]
    settings:
      system_node_pool_machine_type: "n2-standard-8"
      system_node_pool_taints: []
      enable_private_endpoint: false
      enable_pathways_for_tpus: $(vars.enable_pathways_for_tpus)
      enable_gcsfuse_csi: true
      enable_filestore_csi: true   # <-- add this line
      configure_workload_identity_sa: true
      # ... rest of the module unchanged
```

### 3. Deploy the flex-start TPU 7x blueprint

The blueprint provisions the VPCs, the GKE cluster, a TPU 7x node pool with
`enable_flex_start: true` and `auto_repair: false` (a flex-start requirement),
Kueue, and JobSet.

```bash
cd ~/cluster-toolkit
./gcluster deploy examples/gke-consumption-options/dws-flex-start/gke-tpu-7x/gke-tpu-7x.yaml \
  --backend-config="bucket=${TF_STATE_BUCKET}" \
  --vars="project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME},region=${REGION},zone=${ZONE},num_slices=1,machine_type=tpu7x-standard-4t,tpu_topology=${TPU_TOPOLOGY},enable_flex_start=true,autoscaling_min_node_count=0,autoscaling_max_node_count=${MAX_NODES},authorized_cidr=0.0.0.0/0"
```

When prompted, select **(A)pply**. Tighten `authorized_cidr` to your own IP
range (`<ip>/32`) for anything beyond a test. For blueprint details, see the
[DWS flex-start TPU 7x README](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-consumption-options/dws-flex-start/gke-tpu-7x/README.md).

### 4. Connect to the cluster

```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# Only the system node pool is present; the TPU node pool has 0 nodes until a
# workload is submitted.
kubectl get nodes

# Note the Kueue LocalQueue name; it is passed to gcluster job submit below
kubectl get localqueues -n default
export QUEUE=$(kubectl get localqueues -n default -o jsonpath='{.items[0].metadata.name}')
```

## Set up Filestore

1.  Create a Filestore instance on the cluster's primary VPC:

    ```bash
    gcloud filestore instances create "${STORAGE_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --tier=BASIC_HDD \
      --file-share="name=${FILE_SHARE_NAME},capacity=1TB" \
      --network="name=${NETWORK_NAME}"
    ```

1.  Get the instance IP address:

    ```bash
    export FILESTORE_IP=$(gcloud filestore instances describe "${STORAGE_NAME}" \
      --project="${PROJECT_ID}" --zone="${ZONE}" \
      --format="value(networks[0].ipAddresses[0])")
    echo "${FILESTORE_IP}"
    ```

1.  Create the PersistentVolume and PersistentVolumeClaim from
    [`filestore_pv.yaml`](filestore_pv.yaml). The PVC is named
    `filestore-volume`; `gcluster job submit --mount` references it by name.

    ```bash
    sed -e "s#<INSTANCE CAPACITY>#1Ti#g" \
        -e "s#<ZONE>#${ZONE}#g" \
        -e "s#<FILESTORE_INSTANCE_NAME>#${STORAGE_NAME}#g" \
        -e "s#<FILE_SHARE_NAME>#${FILE_SHARE_NAME}#g" \
        -e "s#<INSTANCE IP>#${FILESTORE_IP}#g" \
        filestore_pv.yaml | kubectl apply -f -

    kubectl get pvc filestore-volume -n default   # STATUS should be Bound
    export MOUNT_PATH="/mnt/filestore"
    ```

## Run a workload

> **Flex-start behaviour:** after you submit a job, its pods stay `Pending`
> while Dynamic Workload Scheduler obtains capacity. Watch for a
> `TriggeredScaleUp` event (`kubectl get events -n default`), then nodes join
> and the pods start. Submit with `--restarts 0`; flex-start nodes are reclaimed
> after the job finishes or after a maximum run duration of 7 days.

<details>
<summary><strong>Option A: Mock training workload</strong></summary>

This runs a CPU-only script on the TPU node pool to validate flex-start scale-up
and the Filestore mount. The script writes a marker file to the share.

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
  --mount "filestore-volume;${MOUNT_PATH};rw" \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "python3 -c \"import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/heads/main/examples/fake_training.py', 'fake_training.py')\" && \
python3 fake_training.py && \
echo done > ${MOUNT_PATH}/${WORKLOAD_NAME}-\${JOB_COMPLETION_INDEX:-0}.txt && \
ls -l ${MOUNT_PATH}"
```

In a second terminal, watch the node pool scale from 0:

```bash
kubectl get nodes -w
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
    to the Filestore share.

    ```bash
    export WORKLOAD_NAME="maxtext-$(date +%H%M)"
    export BASE_OUTPUT_DIR="${MOUNT_PATH}/maxtext"

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
      --mount "filestore-volume;${MOUNT_PATH};rw" \
      --verbose \
      --gke-namespace default \
      --name "${WORKLOAD_NAME}" \
      --command "set -e && export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
    python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS}"
    ```

</details>

<details>
<summary><strong>Option C: Train Llama 3.1 70B with MaxText</strong></summary>

Llama 3.1 70B needs at least a `4x4x4` topology (64 chips, `MAX_NODES=16`).
Recreate the cluster with a larger topology if needed, then follow the
[Llama 3.1 70B 4x4x4 Cluster Toolkit recipe](../../../llama3.1-70b/8k-bf16-tpu7x-4x4x4/cluster_toolkit/README.md).
In its `run_recipe.sh`, set `BASE_OUTPUT_DIR` to a path under `${MOUNT_PATH}`
and add `--mount "filestore-volume;${MOUNT_PATH};rw"` to the
`gcluster job submit` command.

</details>

## Monitor the workload

```bash
gcluster job list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
gcluster job logs ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}

# Or with kubectl
kubectl get jobset -n default ${WORKLOAD_NAME}
kubectl get events -n default --sort-by=.lastTimestamp | tail -n 20   # look for TriggeredScaleUp
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default ${POD_NAME}
```

## Clean up

```bash
# Delete a workload
gcluster job cancel ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}

# Remove the PV/PVC, then the Filestore instance (do this before destroying the
# cluster so the VPC can be deleted cleanly)
kubectl delete pvc filestore-volume -n default
kubectl delete pv filestore-pv
gcloud filestore instances delete "${STORAGE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet

# Destroy the cluster and all resources created by the blueprint
cd ~/cluster-toolkit
./gcluster destroy ${CLUSTER_NAME} --auto-approve

# Optionally delete the Terraform state bucket
gcloud storage rm -r "gs://${TF_STATE_BUCKET}"
```

# Run a training workload on Ironwood (TPU v7x) with flex-start and Managed Lustre using Cluster Toolkit

This tutorial walks through an end-to-end flow on Ironwood (Cloud TPU v7x) GKE
clusters by using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`):

1.  Create a GKE cluster whose TPU 7x node pool is provisioned on demand with
    [Dynamic Workload Scheduler flex-start](https://cloud.google.com/kubernetes-engine/docs/how-to/dws-flex-start-training).
    The node pool starts at **0 nodes** and scales up when a workload is
    submitted.
2.  Create a [Managed Lustre](https://cloud.google.com/managed-lustre/docs/overview)
    instance and expose it to the cluster as a shared `ReadWriteMany`
    PersistentVolume.
3.  Run a mock training workload, then a small
    [MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining
    workload, with `gcluster job submit`.

It is the Cluster Toolkit equivalent of the XPK
[flex-start + Lustre recipe](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/tpu7x/recipes/flex_lustre_recipe.md).

For a full model recipe that uses Managed Lustre on a reserved cluster, see
[DeepSeek-V3 671B 4x8x8 (Lustre) with Cluster Toolkit](../../../deepseek3-671b/4k-bf16-tpu7x-4x8x8-lustre/cluster_toolkit/README.md).

## Before you begin

-   A Google Cloud project with billing enabled and access to TPU 7x. Contact
    your account team for Ironwood access.
-   Managed Lustre is available only in
    [specific zones](https://cloud.google.com/managed-lustre/docs/locations);
    pick a `ZONE` that supports both TPU 7x and Managed Lustre.
-   The account you use needs the following IAM roles:
    -   Compute Admin and Compute Network Admin (for private services access)
    -   Kubernetes Engine Admin
    -   Service Account Admin and Service Account User
    -   Storage Admin
    -   Managed Lustre Admin
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
export STORAGE_NAME=<lustre_instance>     # Managed Lustre instance name
export STORAGE_CAPACITY_GIB=18000
export STORAGE_THROUGHPUT=1000            # MB/s per TiB
export STORAGE_FS=lfs
export IP_RANGE_NAME="${CLUSTER_NAME}-lustre-range"
export FIREWALL_RULE_NAME="${CLUSTER_NAME}-allow-lustre"
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

### 2. Enable the Managed Lustre CSI driver in the blueprint

The DWS flex-start TPU 7x blueprint enables the GCS FUSE CSI driver but not the
Managed Lustre CSI driver. Add `enable_managed_lustre_csi: true` to the
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
      enable_managed_lustre_csi: true   # <-- add this line
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

## Set up Managed Lustre

1.  Configure private services access on the cluster's primary VPC. Managed
    Lustre is reached over VPC peering.

    ```bash
    gcloud services enable servicenetworking.googleapis.com lustre.googleapis.com \
      --project="${PROJECT_ID}"

    gcloud compute addresses create "${IP_RANGE_NAME}" \
      --global \
      --purpose=VPC_PEERING \
      --prefix-length=20 \
      --description="Managed Lustre VPC peering" \
      --network="${NETWORK_NAME}" \
      --project="${PROJECT_ID}"

    CIDR_RANGE=$(gcloud compute addresses describe "${IP_RANGE_NAME}" \
      --global \
      --format="value[separator=/](address, prefixLength)" \
      --project="${PROJECT_ID}")

    gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" \
      --allow=tcp:988,tcp:6988 \
      --network="${NETWORK_NAME}" \
      --source-ranges="${CIDR_RANGE}" \
      --project="${PROJECT_ID}"

    # Requires compute.networkAdmin or servicenetworking.networksAdmin
    gcloud services vpc-peerings connect \
      --network="${NETWORK_NAME}" \
      --project="${PROJECT_ID}" \
      --ranges="${IP_RANGE_NAME}" \
      --service=servicenetworking.googleapis.com
    ```

1.  Create the Managed Lustre instance:

    ```bash
    gcloud lustre instances create "${STORAGE_NAME}" \
      --per-unit-storage-throughput="${STORAGE_THROUGHPUT}" \
      --capacity-gib="${STORAGE_CAPACITY_GIB}" \
      --filesystem="${STORAGE_FS}" \
      --location="${ZONE}" \
      --network="projects/${PROJECT_ID}/global/networks/${NETWORK_NAME}" \
      --project="${PROJECT_ID}"
    ```

1.  Get the instance IP address from its `mountPoint` (`<ip>@tcp:/<fs>`):

    ```bash
    export LUSTRE_IP=$(gcloud lustre instances describe "${STORAGE_NAME}" \
      --location="${ZONE}" --project="${PROJECT_ID}" \
      --format="value(mountPoint)" | cut -d'@' -f1)
    echo "${LUSTRE_IP}"
    ```

1.  Create the PersistentVolume and PersistentVolumeClaim from
    [`lustre_pv.yaml`](lustre_pv.yaml). The PVC is named `lustre-volume`;
    `gcluster job submit --mount` references it by name.

    ```bash
    sed -e "s#<INSTANCE CAPACITY>#${STORAGE_CAPACITY_GIB}Gi#g" \
        -e "s#<PROJECT_ID>#${PROJECT_ID}#g" \
        -e "s#<ZONE>#${ZONE}#g" \
        -e "s#<LUSTRE_INSTANCE_NAME>#${STORAGE_NAME}#g" \
        -e "s#<INSTANCE IP>#${LUSTRE_IP}#g" \
        -e "s#<FILESYSTEM NAME>#${STORAGE_FS}#g" \
        lustre_pv.yaml | kubectl apply -f -

    kubectl get pvc lustre-volume -n default   # STATUS should be Bound
    export MOUNT_PATH="/mnt/lustre"
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
and the Lustre mount. The script writes a marker file to the file system.

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
  --mount "lustre-volume;${MOUNT_PATH};rw" \
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
    to the Lustre file system.

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
      --mount "lustre-volume;${MOUNT_PATH};rw" \
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
and add `--mount "lustre-volume;${MOUNT_PATH};rw"` to the
`gcluster job submit` command. The
[DeepSeek-V3 Lustre recipe](../../../deepseek3-671b/4k-bf16-tpu7x-4x8x8-lustre/cluster_toolkit/run_recipe.sh)
shows this pattern.

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

# Remove the PV/PVC, then the Lustre instance and its networking (do this
# before destroying the cluster so the VPC can be deleted cleanly)
kubectl delete pvc lustre-volume -n default
kubectl delete pv lustre-pv
gcloud lustre instances delete "${STORAGE_NAME}" --location="${ZONE}" --project="${PROJECT_ID}" --quiet
gcloud compute firewall-rules delete "${FIREWALL_RULE_NAME}" --project="${PROJECT_ID}" --quiet
gcloud services vpc-peerings delete --network="${NETWORK_NAME}" --project="${PROJECT_ID}" --quiet
gcloud compute addresses delete "${IP_RANGE_NAME}" --global --project="${PROJECT_ID}" --quiet

# Destroy the cluster and all resources created by the blueprint
cd ~/cluster-toolkit
./gcluster destroy ${CLUSTER_NAME} --auto-approve

# Optionally delete the Terraform state bucket
gcloud storage rm -r "gs://${TF_STATE_BUCKET}"
```

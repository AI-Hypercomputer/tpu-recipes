# Run a training workload on Ironwood (TPU v7x) with flex-start and Managed Lustre using Cluster Toolkit

This tutorial walks through an end-to-end flow on Ironwood (Cloud TPU v7x) GKE
clusters by using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`) and the `examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml` blueprint:

1.  Deploy a GKE cluster whose TPU 7x node pool is provisioned on demand with
    [Dynamic Workload Scheduler flex-start](https://cloud.google.com/kubernetes-engine/docs/how-to/dws-flex-start-training).
    The node pool starts at **0 nodes** and scales up when a workload is
    submitted.
2.  Provision Private Service Access (PSA), firewall rules, a
    [Managed Lustre](https://cloud.google.com/managed-lustre/docs/overview)
    instance, and a shared `ReadWriteMany` PersistentVolumeClaim declaratively
    via the blueprint's Managed Lustre modules.
3.  Run a mock training workload, then a small
    [MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining
    workload, with `gcluster job submit`.

For a full model recipe that uses Managed Lustre on a reserved cluster, see
[DeepSeek-V3 671B 4x8x8 (Lustre) with Cluster Toolkit](../../../deepseek3-671b/4k-bf16-tpu7x-4x8x8-lustre/cluster_toolkit/README.md).

## Before you begin

-   A Google Cloud project with billing enabled and access to TPU 7x. Contact
    your account team for Ironwood access.
-   Managed Lustre is available only in
    [specific zones](https://cloud.google.com/managed-lustre/docs/locations);
    pick a `ZONE` that supports both TPU 7x and Managed Lustre.
-   The account you use needs the following IAM roles:
    -   Compute Admin and Compute Network Admin (required for Private Service
        Access VPC peering)
    -   Kubernetes Engine Admin
    -   Project IAM Admin (or Security Admin, to grant roles to the blueprint's
        node pool and workload service accounts)
    -   Service Account Admin and Service Account User
    -   Storage Admin
    -   Managed Lustre Admin
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
>
> With flex-start, the node pool autoscales between 0 and `MAX_NODES`.
> `MAX_NODES` must equal the number of nodes in your slice: total chips in
> `TPU_TOPOLOGY` divided by 4 chips per `tpu7x-standard-4t` node (for example,
> `2x2x1` → `MAX_NODES=1`; `2x2x2` → `MAX_NODES=2`; `4x4x4` → `MAX_NODES=16`).

```bash
export PROJECT_ID=<project_id>            # Your Google Cloud project ID
export ZONE=<zone>                        # Example: us-central1-c (must support TPU 7x and Managed Lustre)
export REGION="${ZONE%-*}"
export CLUSTER_NAME=<cluster_name>        # Unique Cluster Toolkit deployment name (also default Lustre instance ID)
export TPU_TOPOLOGY=<topology>            # Example: 2x2x2
export MAX_NODES=<node_count>             # Example: 2 for a 2x2x2 topology
export LUSTRE_SIZE_GIB=18000              # Storage capacity in GiB (see Managed Lustre performance tiers)
export LUSTRE_THROUGHPUT=1000             # MB/s per TiB (valid values: 125, 250, 500, 1000)
export AUTHORIZED_CIDR="$(curl -s https://ifconfig.me)/32"  # Caller IP for kubectl access (or 0.0.0.0/0 for testing)
export TF_STATE_BUCKET="${PROJECT_ID}-ctk-tf-state"
```

## Create the cluster with flex-start and Managed Lustre

### 1. Enable APIs, create the Terraform state bucket, and authenticate

```bash
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  iam.googleapis.com \
  lustre.googleapis.com \
  servicenetworking.googleapis.com \
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

### 2. Enable the Managed Lustre modules in a copy of `gke-tpu-7x-advanced.yaml`

`examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml` already enables the Managed
Lustre CSI driver (`enable_managed_lustre_csi: true`) on the GKE cluster and
includes commented Managed Lustre variables and four modules under
`# --- MANAGED LUSTRE ADDITIONS ---` (`private_service_access`,
`lustre_firewall_rule`, `managed-lustre`, and `lustre-pv`).

Copy the blueprint, uncomment the Managed Lustre variables and modules, and
ensure `ports:` in `lustre_firewall_rule` is indented under `- protocol: tcp`:

```bash
cd "${HOME}/cluster-toolkit"
cp examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml gke-tpu-7x-lustre.yaml

sed -i \
  -e '/# # Managed-Lustre instance name/,/# per_unit_storage_throughput: 500/{s/^  # /  /}' \
  -e '/# # --- MANAGED LUSTRE ADDITIONS ---/,/#     capacity_gib: \$(vars.lustre_size_gib)/{s/^  # /  /}' \
  -e '/- id: lustre_firewall_rule/,/- id: managed-lustre/{s/^        ports:/          ports:/; s/^        - "988"/          - "988"/}' \
  gke-tpu-7x-lustre.yaml
```

If you edit `gke-tpu-7x-lustre.yaml` manually instead, verify that the
uncommented `lustre_firewall_rule` module matches this structure (`ports` nested
inside the `- protocol: tcp` item):

```yaml
  - id: lustre_firewall_rule
    source: modules/network/firewall-rules
    use: [gke-tpu-7x-net-0]
    settings:
      ingress_rules:
      - name: $(vars.deployment_name)-allow-lustre-traffic
        description: Allow Managed Lustre traffic
        source_ranges:
        - $(private_service_access.cidr_range)
        allow:
        - protocol: tcp
          ports:
          - "988"
```

### 3. Write the flex-start deployment configuration and deploy

Create a deployment file using Option 2 (DWS Flex Start) from
`examples/gke-tpu-7x/gke-tpu-7x-deployment.yaml` along with the Managed Lustre
capacity and throughput variables (see
[Managed Lustre performance tiers](https://cloud.google.com/managed-lustre/docs/create-instance#performance-tiers)
for valid `lustre_size_gib` and `per_unit_storage_throughput` combinations).

```bash
cat <<EOF > gke-tpu-7x-flex-lustre-deployment.yaml
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
  enable_flex_start: true
  autoscaling_min_node_count: 0
  autoscaling_max_node_count: ${MAX_NODES}
  lustre_instance_id: ${CLUSTER_NAME}
  lustre_size_gib: ${LUSTRE_SIZE_GIB}
  per_unit_storage_throughput: ${LUSTRE_THROUGHPUT}
EOF
```

> **System node pool machine type:** If your zone does not offer the default
> `n2d-standard-8` CPU machine type, add
> `system_node_pool_machine_type: n2-standard-8` (or `e2-standard-8`) under
> `vars:`.

Deploy the blueprint (provisioning a Managed Lustre instance typically takes
15–30 minutes):

```bash
./gcluster deploy \
  -d gke-tpu-7x-flex-lustre-deployment.yaml \
  gke-tpu-7x-lustre.yaml
```

When prompted, select **(A)pply**.

### 4. Connect to the cluster and inspect the Managed Lustre PVC

```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# Only the system node pool is present initially; the TPU node pool has 0 nodes
# until a workload is submitted.
kubectl get nodes

# Retrieve the Kueue LocalQueue name (defaults to "user-queue")
kubectl get localqueues -n default
export QUEUE=$(kubectl get localqueues -n default -o jsonpath='{.items[0].metadata.name}')

# Retrieve the Bound Managed Lustre PVC name (defaults to "${CLUSTER_NAME}-pvc")
kubectl get pvc -n default
export LUSTRE_PVC=$(kubectl get pvc -n default \
  -o jsonpath='{.items[?(@.spec.storageClassName=="")].metadata.name}')
export MOUNT_PATH="/lustre"
echo "Managed Lustre PVC: ${LUSTRE_PVC}"
```

## Run a workload

> **Flex-start behavior:** After you submit a job, its pods stay `Pending`
> while Dynamic Workload Scheduler obtains capacity. Watch for a
> `TriggeredScaleUp` event (`kubectl get events -n default`), after which TPU
> nodes join the cluster and pods start. Submit with `--restarts 0`; flex-start
> nodes are reclaimed after the job finishes or after a maximum run duration of
> 7 days.

<details>
<summary><strong>Option A: Mock training workload</strong></summary>

This runs a CPU-only script on the TPU node pool to validate flex-start scale-up
and the Managed Lustre mount. Each container writes a marker file to the Lustre
file system.

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
  --mount "${LUSTRE_PVC};${MOUNT_PATH};rw" \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "python3 -u -c \"import math, time; [time.sleep(0.5) or print(f'Epoch {e}/5 - loss: {2.15 * math.exp(-0.35 * e) + 0.15:.4f}') for e in range(1, 6)]\" && \
echo done > ${MOUNT_PATH}/${WORKLOAD_NAME}-\${TPU_WORKER_ID:-\${JOB_COMPLETION_INDEX:-0}}-\$\$.txt && \
ls -l ${MOUNT_PATH}"
```

In a second terminal, watch the node pool scale up from 0:

```bash
kubectl get nodes -w
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
    to the Managed Lustre file system.

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
      --service-account workload-identity-k8s-sa \
      --image "${WORKLOAD_IMAGE}" \
      --mount "${LUSTRE_PVC};${MOUNT_PATH};rw" \
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
In its `run_recipe.sh`, set `BASE_OUTPUT_DIR` to a path under `${MOUNT_PATH}`,
set `--queue "${QUEUE}"`, and add `--service-account workload-identity-k8s-sa`
and `--mount "${LUSTRE_PVC};${MOUNT_PATH};rw"` to `gcluster job submit`. The
[DeepSeek-V3 Lustre recipe](../../../deepseek3-671b/4k-bf16-tpu7x-4x8x8-lustre/cluster_toolkit/run_recipe.sh)
shows how POSIX log copying (`mkdir -p ${ARTIFACT_DIR}/logs && cp train.log ...`)
is structured on a Lustre mount.

</details>

## Monitor the workload

```bash
gcluster job list --cluster "${CLUSTER_NAME}" --project "${PROJECT_ID}" --location "${ZONE}"
gcluster job logs "${WORKLOAD_NAME}" --cluster "${CLUSTER_NAME}" --project "${PROJECT_ID}" --location "${ZONE}"

# Or with kubectl
kubectl get jobset -n default "${WORKLOAD_NAME}"
kubectl get events -n default --sort-by=.lastTimestamp | tail -n 20   # look for TriggeredScaleUp
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${WORKLOAD_NAME}" -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default "${POD_NAME}"
```

## Clean up

Because Managed Lustre, its PersistentVolumeClaim, the Lustre firewall rule, and
Private Service Access VPC peering are all managed by Terraform in the
blueprint, `gcluster destroy` deletes them and the GKE cluster in dependency
order automatically:

```bash
# Cancel an active workload (if still running)
gcluster job cancel "${WORKLOAD_NAME}" --cluster "${CLUSTER_NAME}" --project "${PROJECT_ID}" --location "${ZONE}"

# Destroy the cluster, Managed Lustre instance, PSA peering, PVCs/PVs, and networking
cd "${HOME}/cluster-toolkit"
./gcluster destroy "${CLUSTER_NAME}" --auto-approve

# Optionally delete the Terraform state bucket
gcloud storage rm -r "gs://${TF_STATE_BUCKET}"
```

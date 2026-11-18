# Pretrain llama3.1-70b workload on Cloud TPU 7x (Ironwood) GKE clusters with Cluster Toolkit (gcluster)

This recipe outlines the steps for running a `llama3.1-70b`
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on an
[Ironwood GKE cluster](https://cloud.google.com/kubernetes-engine)
by using
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).

Set the environment variables in [Environment variables](#environment-variables)
first; every command below reads them, so the same values are used for
provisioning, submission and teardown.

## Workload details

-   **Model:** `llama3.1-70b`
-   **Sequence length:** 131072
-   **Precision:** `fp8_full`
-   **Global batch size:** 64 (`per_device_batch_size=0.125`, `ici_context_parallelism=8`)
-   **Accelerator:** `tpu7x-4x8x8` (256 chips)
-   **Slices:** 1
-   **Topology:** `4x8x8`

## Prerequisites

To run this recipe, you need the following:

-   **GCP project setup:** a project with billing enabled, allowlisted for
    Ironwood access.
-   **IAM roles** on that project:
    -   Compute Admin
    -   Kubernetes Engine Admin
    -   Logging Admin
    -   Monitoring Admin
    -   Service Account User
    -   Storage Admin
    -   Service Usage Consumer
-   **Cluster Toolkit (`gcluster`) and dependencies:** see
    [Install Cluster Toolkit and dependencies](#install-cluster-toolkit-and-dependencies).

## Install Cluster Toolkit and dependencies

Follow the official
[Cluster Toolkit environment setup guide](https://docs.cloud.google.com/cluster-toolkit/docs/setup/configure-environment#install)
to configure your environment.

### Cluster Toolkit (gcluster)

Install Cluster Toolkit by downloading and extracting the prebuilt release
bundle:

```bash
export CTK_VERSION="1.108.0"

curl -L -O "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/v${CTK_VERSION}/gcluster_bundle_linux_amd64.tgz"

mkdir -p "${HOME}/cluster-toolkit"
tar -xzf gcluster_bundle_linux_amd64.tgz -C "${HOME}/cluster-toolkit"
rm gcluster_bundle_linux_amd64.tgz

export PATH="${HOME}/cluster-toolkit:${PATH}"
echo 'export PATH="${HOME}/cluster-toolkit:${PATH}"' >> ~/.bashrc

gcluster --version
```

### gcloud, kubectl, and the auth plugin

```bash
# Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install
# Install kubectl: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl
# Install gke-gcloud-auth-plugin: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin

gcloud auth login
gcloud auth application-default login
```

### Docker

Install Docker and configure permissions for your user and Artifact Registry:

```bash
# Add your user to the docker group (re-login after running this):
sudo usermod -aG docker "${USER}"

# Configure Docker to authenticate with Google Cloud Artifact Registry:
gcloud auth configure-docker
```

## Orchestration and deployment tools

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Job configuration and deployment** -
    [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
    (`gcluster`) deploys the
    [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    that runs the workload.

## Environment variables

```bash
export PROJECT_ID="<your-project-id>"
export CLUSTER_NAME="<your-cluster-name>"
export ZONE="<your-zone>"              # for example, us-central1-c
export REGION="${ZONE%-*}"             # the cluster is regional; ZONE pins its node pools
export NAMESPACE="default"
export KUEUE_QUEUE=""                  # empty: gcluster auto-discovers the LocalQueue
export WORKLOAD_NAME="$(echo "${USER}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -cd 'a-z0-9-' | sed 's/^-*//' | cut -c1-10)-benchmark-$(date +%H%M)"
export WORKLOAD_IMAGE="<your-container-image>"
export BUCKET_NAME="<your-gcs-bucket>"
export BASE_OUTPUT_DIR="gs://${BUCKET_NAME}"
export ARTIFACT_DIR="${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}"
export RESERVATION_NAME="<your-reservation-name>"
```

-   `RESERVATION_NAME`: the bare reservation name (for example,
    `my-reservation`). For shared reservations in another project, also
    uncomment and set `project` in the cluster deployment file.
-   `KUEUE_QUEUE` is the Kueue LocalQueue the workload is submitted to. Leave it
    empty on a cluster created from the blueprint below: it installs a single
    LocalQueue (`user-queue`) in `${NAMESPACE}`, which `gcluster job submit`
    discovers on its own. Set it only when the namespace holds several
    LocalQueues, for example `multislice-queue` on a cluster provisioned with
    XPK.
-   `BUCKET_NAME` is the GCS bucket the workload writes to. Storage mounts and
    paths in `run_recipe.sh` that pointed at the original run's
    bucket read it (or `BASE_OUTPUT_DIR`) instead.
-   `ARTIFACT_DIR` is passed to the workload as an environment variable. If the
    bucket does not exist yet, create it:

    ```bash
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
      --project="${PROJECT_ID}" \
      --location=US \
      --default-storage-class=STANDARD \
      --uniform-bucket-level-access
    ```

## Build the workload container image

### Steps for building workload image

The following software versions were used to validate this recipe:

-   MaxText commit: `5651d7f`
-   JAX version: `0.12.1.dev20261103`
-   Libtpu version: `0.0.51`
-   Python: `3.12`
-   Cluster Toolkit: `1.108.0`

The recipe runs the container image named by `WORKLOAD_IMAGE`. To build a
MaxText image from the validated versions, set `WORKLOAD_IMAGE` to an Artifact
Registry path you can push to, for example
`us-docker.pkg.dev/<your-project-id>/<your-repository>/maxtext-runner:latest`,
then build the image and push it to that path:

```bash
gcloud config set project "${PROJECT_ID}"

git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout 5651d7f

bash src/dependencies/scripts/docker_build_dependency_image.sh \
  MODE=nightly \
  DEVICE=tpu \
  JAX_VERSION=0.12.1.dev20261103 \
  LIBTPU_VERSION=0.0.51
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME="${WORKLOAD_IMAGE}"

cd ..
```

## Create the cluster

Cluster Toolkit uses blueprints and deployment configuration files to provision
GKE clusters with Cloud TPU node pools. It calculates the number of nodes
required from `tpu_topology` and `compute_machine_type`. For an overview of
topologies, slices and blueprint options, see the
[Cloud TPU deployments overview](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview).

For detailed instructions and configuration options for
Cloud TPU 7x (Ironwood), see the
[Cloud TPU 7x (Ironwood) GKE deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x#deploy-tpu-7x-cluster).

### 1. Set up the Terraform state bucket

Create a Cloud Storage bucket to hold the Terraform state for the deployment,
enable versioning on it, and generate Application Default Credentials so that
Terraform can authenticate:

```bash
export TF_STATE_BUCKET="${PROJECT_ID}-ctk-tf-state"

gcloud storage buckets create "gs://${TF_STATE_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access

gcloud storage buckets update "gs://${TF_STATE_BUCKET}" --versioning

gcloud auth application-default login
```

### 2. Deploy the cluster

Write the deployment configuration file `examples/gke-tpu-7x/gke-tpu-7x-deployment.yaml` from the
variables you exported:

```bash
cd ~/cluster-toolkit
cat > examples/gke-tpu-7x/gke-tpu-7x-deployment.yaml <<EOF
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
  tpu_topology: 4x8x8
  authorized_cidr: 0.0.0.0/0
  # Namespace that receives the Kueue LocalQueue the blueprint installs.
  user_namespace: ${NAMESPACE}
  # For on-demand deployments (no reservation), omit or comment out reservation_affinity below.
  reservation_affinity:
    consume_reservation_type: SPECIFIC_RESERVATION
    specific_reservations:
    - name: ${RESERVATION_NAME}
      # project: <reservation-project-id>  # Uncomment for shared reservations
EOF
```

and deploy using it:

```bash
./gcluster deploy -d \
  examples/gke-tpu-7x/gke-tpu-7x-deployment.yaml \
  examples/gke-tpu-7x/gke-tpu-7x.yaml
```

The blueprint installs Kueue with a `LocalQueue` named `user-queue` in
`user_namespace`. `gcluster job submit` discovers it automatically when the
namespace has exactly one `LocalQueue`.

To inspect the generated Terraform configuration before applying it, use
`gcluster create`:

```bash
./gcluster create examples/gke-tpu-7x/gke-tpu-7x.yaml \
  -d examples/gke-tpu-7x/gke-tpu-7x-deployment.yaml \
  -o deployment/
./gcluster deploy deployment/
```

### 3. Advanced blueprint (optional)

For workloads that need a multi-VPC network, pre-configured Cloud Storage FUSE
mounts, Hyperdisk Balanced, Managed Lustre or Filestore, use the advanced
blueprint `examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml` with the same deployment file (see
[Advanced blueprints](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview#advanced-blueprints)):

```bash
cd ~/cluster-toolkit
./gcluster deploy -d \
  examples/gke-tpu-7x/gke-tpu-7x-deployment.yaml \
  examples/gke-tpu-7x/gke-tpu-7x-advanced.yaml
```

### Connect to your cluster

Once deployment is complete, fetch credentials and confirm the nodes are ready:

```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}"

kubectl get nodes
```

A cluster deployed from the blueprint above already has `${NAMESPACE}` (its
`user_namespace`) and the LocalQueue in it. On a cluster you brought yourself,
create the namespace if it does not already exist:

```bash
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
```

## Run the recipe

`run_recipe.sh` carries the full `gcluster job submit` command
for this workload, including its accelerator, topology, storage mounts and
environment. It reuses the variables you exported above, so you can run it as
is:

```bash
git clone https://github.com/AI-Hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/llama3.1-70b/128k-fp8-tpu7x-4x8x8/cluster_toolkit
chmod +x run_recipe.sh
./run_recipe.sh
```

The workload command is inlined into `run_recipe.sh` under
`--command`, so make any edits directly in that script.
`run_workload.sh` is provided as a standalone reference copy.

`gcluster job submit` creates a JobSet and, on a cluster using Kueue, enqueues
it for admission. The command returns as soon as the workload is accepted, so a
successful submission does not by itself mean the workload is running.

`../k8s/k8s_manifest.yaml` is the Kubernetes JobSet the submission expands to, for
review or for applying with `kubectl` directly:

```bash
envsubst '${ARTIFACT_DIR} ${BASE_OUTPUT_DIR} ${WORKLOAD_IMAGE} ${WORKLOAD_NAME}' \
  < ../k8s/k8s_manifest.yaml | kubectl apply -n "${NAMESPACE}" -f -
```

## Monitor the job

Stream the logs:

```bash
gcluster job logs "${WORKLOAD_NAME}" \
  --cluster "${CLUSTER_NAME}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --gke-namespace "${NAMESPACE}" \
  --follow
```

Or inspect the JobSet directly:

```bash
kubectl get jobset "${WORKLOAD_NAME}" -n "${NAMESPACE}"

kubectl get pods -n "${NAMESPACE}" \
  -l jobset.sigs.k8s.io/jobset-name="${WORKLOAD_NAME}"

# Scheduling events, useful when pods stay Pending.
kubectl describe jobset "${WORKLOAD_NAME}" -n "${NAMESPACE}"
```

Logs are also retained in Cloud Logging, which is the more reliable source once
the pods have been reclaimed:

```text
resource.type="k8s_container"
resource.labels.cluster_name="${CLUSTER_NAME}"
resource.labels.namespace_name="${NAMESPACE}"
labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"="${WORKLOAD_NAME}"
```

## Results and artifacts

The workload writes its outputs to `${ARTIFACT_DIR}`:

```bash
gcloud storage ls --recursive "${ARTIFACT_DIR}"
```

## Clean up

Cancel the workload and release its resources:

```bash
gcluster job cancel "${WORKLOAD_NAME}" \
  --cluster "${CLUSTER_NAME}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --gke-namespace "${NAMESPACE}"
```

Cancelling does not delete anything already written to `${ARTIFACT_DIR}`.

To tear down the cluster itself:

```bash
cd ~/cluster-toolkit
./gcluster destroy "${CLUSTER_NAME}"
```

## Troubleshooting

-   **`gcluster: command not found`.** The Cluster Toolkit CLI is not on your
    `PATH`. See
    [Install Cluster Toolkit and dependencies](#install-cluster-toolkit-and-dependencies).
-   **Pods stay `Pending`.** The cluster usually has no free capacity of the
    requested accelerator type, or Kueue has not admitted the workload. Check
    `kubectl describe jobset "${WORKLOAD_NAME}" -n "${NAMESPACE}"` for the
    scheduling events.
-   **The submission is rejected with a namespace error.** Confirm
    `${NAMESPACE}` exists and that you can write to it.
-   **No logs appear.** Until the first pod starts there is nothing to stream.
    Confirm the workload was admitted before investigating the logging path.

# Pretrain qwen3-235b-a22b workload on Ironwood GKE clusters with Cluster Toolkit

This recipe outlines the steps for running a qwen3-235b-a22b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).


## Workload Details

This workload is configured with the following details:

-   Sequence Length: 4096
-   Precision: bfloat16
-   Chips: 64 (4x8x8 topology)

## Prerequisites

To run this recipe, you need the following:

-   **GCP Project Setup:** Ensure you have a GCP project with billing enabled
    and are allowlisted for Ironwood access.
-   **User Project Permissions:** The account used requires the following IAM
    Roles:
    -   Artifact Registry Writer
    -   Compute Admin
    -   Kubernetes Engine Admin
    -   Logging Admin
    -   Monitoring Admin
    -   Service Account User
    -   Storage Admin
    -   Vertex AI Administrator
    -   Service Usage Consumer
    -   TPU Viewer
-   **Docker:** Docker must be installed on your workstation. Follow the steps
    in the
    [Install Cluster Toolkit and dependencies](#install-cluster-toolkit-and-dependencies)
    section to install Docker.
-   **Cluster Toolkit (gcluster) and Dependencies:** Follow the steps in the
    [Install Cluster Toolkit and dependencies](#install-cluster-toolkit-and-dependencies)
    section to install Cluster Toolkit (`gcluster`), `gcloud`, `kubectl`, and
    the `gke-gcloud-auth-plugin`.


## Install Cluster Toolkit and dependencies

### Cluster Toolkit (gcluster)

Install Cluster Toolkit by downloading and extracting the prebuilt release
bundle:

```bash
# Set Cluster Toolkit version
export CTK_VERSION="1.104.0"

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

### Tools (gcloud, kubectl, and auth plugin)

```bash
# Install Google Cloud SDK (gcloud): https://cloud.google.com/sdk/docs/install
# Install kubectl: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl
# Install gke-gcloud-auth-plugin: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin

# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

### Docker

Install Docker using instructions provided by your administrator. Once
installed, run the following commands:

```bash
## Configure docker and test installation
gcloud auth configure-docker
sudo usermod -aG docker $USER ## relaunch the terminal after running this command
docker run hello-world # Test docker
```


## Orchestration and deployment tools

For this recipe, the following setup is used:

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Pretraining job configuration and deployment** -
    [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
    (`gcluster`) is used to configure and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the qwen3-235b-a22b workload.


## Test environment

This recipe is optimized for and tested with tpu7x-4x8x8.

-   **GKE cluster** To create your GKE cluster, refer to the
    [Cloud TPU deployments overview](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview)
    and the [Cloud TPU 7x (Ironwood) GKE deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x#deploy-tpu-7x-cluster).
    A sample Cluster Toolkit cluster creation and deployment command is provided below.

### Environment Variables for Cluster Creation

The environment variables required for cluster creation and workload execution
are defined at the beginning of the `run_recipe.sh` script. **Before running the
`gcluster job submit` command**, please open `run_recipe.sh` and modify the
`export` statements to set these variables to match your environment. It is
crucial to use consistent values for `PROJECT_ID`, `CLUSTER_NAME`, and `ZONE`
across all commands and configurations.

-   `PROJECT_ID`: Your GCP project name.
-   `CLUSTER_NAME`: The target cluster name.
-   `ZONE`: The zone for your cluster (e.g., `us-central1-c`).
-   `REGION`: The region for your cluster (e.g., `us-central1`). Can be derived as `${ZONE%-*}`.
-   `CONTAINER_REGISTRY`: The container registry to use (e.g., `gcr.io`).
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `WORKLOAD_IMAGE`: The Docker image for the workload. This is set in
    `run_recipe.sh` to
    `${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-qwen3-235b-a22b-runner` by
    default, matching the image built in the
    [Docker container image](#docker-container-image) section.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` to `${USER}-qwen3-235b-a22b-$(date +%H%M)` by default.
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `ACCELERATOR_TYPE`: The TPU type (e.g., `tpu7x-4x8x8`). See topologies
    [here](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#configuration).
-   `RESERVATION_NAME`: Your TPU reservation name. Use the reservation name if
    within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

If you don't have a GCS bucket, create one with this command:

```bash
# Make sure BASE_OUTPUT_DIR is set in run_recipe.sh before running this.
gcloud storage buckets create ${BASE_OUTPUT_DIR} --project=${PROJECT_ID} --location=US  --default-storage-class=STANDARD --uniform-bucket-level-access
```

### Sample Cluster Toolkit Cluster Creation and Deployment Command

Cluster Toolkit uses blueprints and deployment configurations to provision GKE
clusters with Cloud TPU node pools. For detailed deployment instructions and configuration options for Cloud TPU 7x (Ironwood),
refer to the [Cloud TPU 7x (Ironwood) GKE deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x#deploy-tpu-7x-cluster).

#### 1. Set up Terraform State Bucket and Authentication

```bash
export TF_STATE_BUCKET="${PROJECT_ID}-ctk-tf-state"
export REGION="${ZONE%-*}"

# Create bucket to store Terraform state
gcloud storage buckets create "gs://${TF_STATE_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access

# Enable versioning on the bucket
gcloud storage buckets update "gs://${TF_STATE_BUCKET}" --versioning

# Generate Application Default Credentials for Terraform
gcloud auth application-default login
```

#### 2. Deploy Cluster with Cluster Toolkit

Deploy the standard blueprint to provision the GKE infrastructure using
`gcluster deploy`:

```bash
cd ~/cluster-toolkit
./gcluster deploy examples/gke-tpu-7x/gke-tpu-7x.yaml \
  --backend-config="bucket=${TF_STATE_BUCKET}" \
  --vars="project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME},region=${REGION},zone=${ZONE},num_slices=1,machine_type=tpu7x-standard-4t,tpu_topology=4x8x8,reservation=${RESERVATION_NAME}"
```

#### 3. Connect to Your Cluster

Once deployment is complete, fetch credentials to configure `kubectl` access and
verify that the cluster and TPU nodes are ready:

```bash
# Connect to your cluster and configure kubectl credentials
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# Verify that cluster nodes are in Ready state
kubectl get nodes
```


## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section above for installing
Cluster Toolkit and its dependencies. Docker installation is part of this
process.

### Steps for building workload image

The following software versions are used:

-   Libtpu version: 0.0.46
-   Jax version: 0.11.2.dev20260825
-   Maxtext version: 9d92bf0
-   Python: 3.12
-   Cluster Toolkit: 1.104.0

Docker Image Building Command:

```bash
export CONTAINER_REGISTRY="" # Initialize with your registry
export CLOUD_IMAGE_NAME="${USER}-maxtext-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

# Set up and Activate Python 3.12 virtual environment for Docker build
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.12 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

# Make sure you're running on a Virtual Environment with python 3.12
if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" == "3.12" ]]; then { echo "You have the correct Python version 3.12"; } else { >&2 echo "Error: Python version must be 3.12."; false; } fi

# Clone MaxText Repository and Checkout Recipe Branch
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout 9d92bf0

# Build and upload the docker image
bash src/dependencies/scripts/docker_build_dependency_image.sh \
  MODE=nightly \
  JAX_VERSION=0.11.2.dev20260825 \
  LIBTPU_VERSION=0.0.46
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

# Deactivate the virtual environment
deactivate

# Return to the recipe directory
cd ..
```

## Training dataset

This recipe uses a mock pretraining dataset provided by the MaxText framework.

## Run the recipe

### Configure environment settings

Before running any commands in this section, ensure you have set the environment
variables as described in
[Environment Variables for Cluster Creation](#environment-variables-for-cluster-creation).

### Connect to an existing cluster (Optional)

If you want to connect to your GKE cluster to see its current state before
running the benchmark, you can use the following gcloud command:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

### Run qwen3-235b-a22b Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the qwen3-235b-a22b pretraining workload.

To run the benchmark, first make the script executable, edit it to configure
environment variables, and then run it:

```bash
chmod +x run_recipe.sh
nano run_recipe.sh
./run_recipe.sh
```

You can customize the run by modifying `run_recipe.sh`:

-   **Environment Variables:** Variables like `PROJECT_ID`, `CLUSTER_NAME`,
    `ZONE`, `WORKLOAD_NAME`, `WORKLOAD_IMAGE`, and `BASE_OUTPUT_DIR` are defined
    at the beginning of the script. Adjust these to match your environment.
-   **XLA Flags:** The `XLA_FLAGS` variable contains a set of XLA configurations
    optimized for this workload. These can be tuned for performance or
    debugging.
-   **MaxText Workload Overrides:** The `MAXTEXT_ARGS` variable holds the
    arguments passed to the `python3 -m maxtext.trainers.pre_train.train`
    command. This includes model-specific settings like `per_device_batch_size`,
    `max_target_length`, and others. You can modify these to experiment with
    different model configurations.

Note that any MaxText configurations not explicitly overridden in `MAXTEXT_ARGS`
are expected to use the defaults within the specified `WORKLOAD_IMAGE`.

## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and logs:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# Get the name of the first pod in the JobSet
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')

# Follow the logs of that pod
kubectl logs -f -n default ${POD_NAME}
```

You can also monitor your cluster and TPU usage through the Google Cloud
Console:
`https://console.cloud.google.com/kubernetes/workload/overview?project=${PROJECT_ID}`

### Follow Workload and View Metrics

List workloads using Cluster Toolkit (`gcluster job list`):

```bash
gcluster job list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

For more in-depth debugging, inspect the workload with `gcluster job inspect`:

```bash
gcluster job inspect --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE} --name ${WORKLOAD_NAME}
```

View workload logs with `gcluster job logs`:

```bash
gcluster job logs ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

### Delete resources

#### Delete a specific workload

To cancel and delete the workload using Cluster Toolkit:

```bash
gcluster job cancel ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

Or delete the JobSet directly using kubectl:

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default
```

#### Delete the entire cluster

To avoid recurring charges, destroy the cluster infrastructure provisioned by
Cluster Toolkit:

```bash
cd ~/cluster-toolkit
./gcluster destroy ${CLUSTER_NAME} --auto-approve
```

## Check results

After the job completes, you can check the results by:

-   Accessing output logs from your job using `kubectl logs` or `gcluster job
    logs`.
-   Checking any data stored in the Google Cloud Storage bucket specified by the
    `${BASE_OUTPUT_DIR}` variable in your `run_recipe.sh`.
-   Reviewing metrics in Cloud Monitoring, if configured.


## Next steps: deeper exploration and customization

This recipe is designed to provide a simple, reproducible "0-to-1" experience
for running a MaxText pre-training workload. Its primary purpose is to help you
verify your environment and achieve a first success with TPUs quickly and
reliably.

For deeper exploration, including customizing model configurations, tuning
performance with different XLA flags, and running custom experiments, we
recommend using the benchmark_runner.py script directly from the MaxText
repository. This script offers the full range of MaxText's flexibility and is
the ideal tool for power users and researchers who want to move beyond the
initial benchmark and tailor the workload to their specific needs. To learn
more, see the
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md)
on using benchmark_runner.py for advanced benchmarking.

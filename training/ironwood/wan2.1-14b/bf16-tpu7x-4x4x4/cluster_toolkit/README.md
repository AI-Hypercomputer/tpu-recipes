# Pretrain wan2.1-14b workload on Ironwood GKE clusters with Cluster Toolkit

<!-- disableFinding(LINK_ID) -->

This recipe outlines the steps for running a wan2.1-14b
[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) pretraining
workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).

## Prerequisites

<!-- disableFinding(LINK_ID) -->

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
# Install gcloud, if not already installed, https://cloud.google.com/sdk/docs/install
# Install kubectl, if not already installed, https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl
# Follow https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin to install gke-gcloud-auth-plugin

# Ensure to log in to your gcloud
gcloud auth login
gcloud auth application-default login

# Set Cluster Toolkit version
export CTK_VERSION="1.104.0"

# Download and extract the prebuilt gcluster binary
mkdir -p ${HOME}/cluster-toolkit
curl -Lo /tmp/gcluster_bundle.tgz \
  https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/v${CTK_VERSION}/gcluster_bundle_linux_amd64.tgz
tar -xzf /tmp/gcluster_bundle.tgz -C ${HOME}/cluster-toolkit gcluster
rm -f /tmp/gcluster_bundle.tgz
chmod +x ${HOME}/cluster-toolkit/gcluster
export PATH="${HOME}/cluster-toolkit:${PATH}"

# Verify installation
gcluster --version
```

#### Docker

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
    resource, which manages the execution of the wan2.1-14b workload.


## Test environment

<!-- disableFinding(LINK_ID) -->

This recipe is optimized for and tested with tpu7x-4x4x4.

-   **GKE cluster** To create your GKE cluster, use Cluster Toolkit (`gcluster`).
    A sample Cluster Toolkit cluster creation and deployment command is provided
    below.

### Environment Variables for Cluster Creation

The environment variables required for cluster creation and workload execution
are defined at the beginning of the `run_recipe.sh` script. **Before running
`run_recipe.sh`**, please open `run_recipe.sh` and modify the `export`
statements to set these variables to match your environment. It is crucial to
use consistent values for `PROJECT_ID`, `CLUSTER_NAME`, and `ZONE` across all
commands and configurations.

-   `PROJECT_ID`: Your GCP project name.
-   `CLUSTER_NAME`: The target cluster name.
-   `ZONE`: The zone for your cluster (e.g., `us-central1-c`).
-   `CONTAINER_REGISTRY`: The container registry to use (e.g., `gcr.io`).
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `WORKLOAD_IMAGE`: The Docker image for the workload. This should be set in
    `run_recipe.sh` to `gcr.io/${PROJECT_ID}/${USER}-maxdiffusion-runner`,
    matching the image built in the
    [Docker container image](#docker-container-image) section.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` to `$(printf "%.11s" "${USER//_/-}")-wan2-1-$(date +%H%M)`
    by default (Cluster Toolkit limits workload names to 28 characters).
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `ACCELERATOR_TYPE`: The TPU machine type and topology (`tpu7x` with topology
    `4x4x4`). See topologies
    [here](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#configuration).
-   `RESERVATION_NAME`: Your TPU reservation name. Use the reservation name if
    within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

If you don't have a GCS bucket, create one with this command:

```bash
# Make sure BASE_OUTPUT_DIR is set in run_recipe.sh before running this.
gcloud storage buckets create ${BASE_OUTPUT_DIR} --project=${PROJECT_ID} --location=US --default-storage-class=STANDARD --uniform-bucket-level-access
```

### Sample Cluster Toolkit Cluster Creation Command

```bash
export PROJECT_ID=<YOUR_PROJECT_ID>
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export REGION=<YOUR_REGION> # e.g., us-central1
export ZONE=<YOUR_ZONE> # e.g., us-central1-c
export RESERVATION_NAME=<YOUR_RESERVATION_NAME>
export TF_STATE_BUCKET="${PROJECT_ID}-cluster-toolkit-state"

# Create a GCS bucket for Terraform state (if not already created)
gcloud storage buckets create gs://${TF_STATE_BUCKET} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --uniform-bucket-level-access

# Deploy the TPU 7x GKE blueprint using Cluster Toolkit
gcluster deploy examples/gke-tpu-7x/gke-tpu-7x.yaml \
  --backend-config="bucket=${TF_STATE_BUCKET}" \
  --vars="project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME},region=${REGION},zone=${ZONE},num_slices=1,machine_type=tpu7x-standard-4t,tpu_topology=4x4x4,reservation=${RESERVATION_NAME}"
```


## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section above for installing
Cluster Toolkit and its dependencies. Docker installation is part of this
process.

### Steps for building workload image

The following software versions are used:

-   Python: 3.12
-   Cluster Toolkit: 1.104.0

Docker Image Building Command:

```bash
# Check if USER is set correctly
export CLOUD_IMAGE_NAME="${USER}-maxdiffusion-runner"
export WORKLOAD_IMAGE="gcr.io/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"
gcloud config set project $PROJECT_ID

# Set up and Activate Python 3.12 virtual environment for Docker build
uv venv --seed ~/.local/bin/venv-docker --python 3.12 --clear
source ~/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

# Make sure you're running on a Virtual Environment with python 3.12
[[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" == "3.12" ]] || { >&2 echo "Error: Python version must be 3.12."; false; }

# Change to your home directory and clone maxdiffusion
cd ~/
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion
git checkout v3

# Run WAN 2.1 Docker build
export RECIPE_DOCKER_IMAGE=maxdiffusion_base_image
bash docker_build_dependency_image.sh mode=nightly
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

# Deactivate the virtual environment
deactivate
```

## Training dataset

Before training, prepare the video training dataset. For this example, we will
be using the PusaV1 dataset.

```bash
# Set to TPU or CPU based on your local setup
# bash setup.sh MODE=stable DEVICE=cpu
bash setup.sh MODE=stable DEVICE=tpu
source ${HOME}/.local/bin/venv/bin/activate

# Following assumes that you have mounted an external drive to your VM
# and you have created a mount point for `/mnt/disks/external_disk`
# along with given write permissions to the directories described below.
export HF_TOKEN=<token>
export HF_DATASET_DIR=/mnt/disks/external_disk/PusaV1_training/
export TFRECORDS_DATASET_DIR=/mnt/disks/external_disk/wan_tfr_dataset_pusa_v1
export HF_HUB_CACHE=/mnt/disks/external_disk/maxdiffusion_hf_cache/
export DATASET_DIR=${BASE_OUTPUT_DIR}/PusaV1_training

# Download the dataset
huggingface-cli download RaphaelLiu/PusaV1_training --repo-type dataset --local-dir ${HF_DATASET_DIR}

# Preprocess and convert the dataset to TFRecord format
# This can be done outside of Cluster Toolkit, or a TPU machine. We recommend using a CPU machine for this preprocessing step. Use skip_jax_distributed_system=True when running on cpu machine.
python src/maxdiffusion/data_preprocessing/wan_pusav1_to_tfrecords.py src/maxdiffusion/configs/base_wan_14b.yml train_data_dir=${HF_DATASET_DIR} tfrecords_dir=${TFRECORDS_DATASET_DIR} no_records_per_shard=10 skip_jax_distributed_system=True

# Upload to gcs
gcloud storage cp --recursive ${TFRECORDS_DATASET_DIR} ${DATASET_DIR}
```

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

## Get the recipe
```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/wan2.1-14b/bf16-tpu7x-4x4x4/cluster_toolkit
```

### Run wan2.1-14b Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the wan2.1-14b pretraining workload.

Before execution, use `nano ./run_recipe.sh` to edit the script and configure the environment variables to match your specific environment.

To configure and run the benchmark:

```bash
chmod +x run_recipe.sh
nano ./run_recipe.sh
./run_recipe.sh
```

You can customize the run by modifying `run_recipe.sh`:

-   **Environment Variables:** Variables like `PROJECT_ID`, `CLUSTER_NAME`,
    `ZONE`, `WORKLOAD_NAME`, `WORKLOAD_IMAGE`, and `BASE_OUTPUT_DIR` are defined
    at the beginning of the script. Adjust these to match your environment.
-   **MaxDiffusion Workload Overrides:** The `MAXDIFFUSION_ARGS` variable holds
    the arguments passed to the training command. This includes model-specific
    settings.

Note that any MaxDiffusion configurations not explicitly overridden in
`MAXDIFFUSION_ARGS` are expected to use the defaults within the specified
`WORKLOAD_IMAGE`.

## Monitor the job

To monitor your job's progress, you can use `kubectl` to check the JobSet status
and stream logs:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# Get the name of the first pod in the JobSet
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')

# Follow the logs of that pod
kubectl logs -f -n default ${POD_NAME}
```
You can also monitor your cluster and TPU usage through the Google Cloud
Console.

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

Or delete the JobSet directly using `kubectl`:

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

This recipe provides a starting point for running MaxDiffusion workloads. For
advanced usage, including exploring different models, datasets, and training
parameters, please refer to the
[MaxDiffusion GitHub repository](https://github.com/AI-Hypercomputer/maxdiffusion).

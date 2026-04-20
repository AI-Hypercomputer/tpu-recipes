# Instructions for training DeepSeek3-671B on TPU Ironwood (tpu7x-4x8x8) with Google Cloud Storage (GCS)

This recipe outlines the steps for running a deepseek3-671b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[XPK](https://github.com/AI-Hypercomputer/xpk) with Google Cloud Storage (GCS) configured as the primary storage system for the dataset and checkpoints.

## Workload Details

This workload is configured with the following details:

-   Sequence Length: 4096
-   Precision: bf16
-   Chips: 256 (4x8x8 topology)
-   GCS buckets for dataset and checkpoints
    -   C4 Multi-Lingual dataset (~12TB) with ArrayRecord format

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
    in the [Install XPK and dependencies](#install-xpk-and-dependencies) section
    to install Docker.
-   **Python 3.11 Virtual Environment:** A Python
    3.11 virtual environment is required. Instructions
    for setting this up are also in the
    [Install XPK and dependencies](#install-xpk-and-dependencies) section.
-   **XPK and Dependencies:** Follow the steps in the
    [Install XPK and dependencies](#install-xpk-and-dependencies) section to
    install XPK, `kubectl`, `kubectl-kueue`, and `kubectl-kjob`.

## Install XPK and dependencies

### XPK and Dependency Installation

#### Virtual Python Environment

Run the following to create a virtual Python environment:

```bash
# Set up uv
sudo apt update
curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh
chmod +x install-uv.sh
./install-uv.sh
rm install-uv.sh
source ${HOME}/.local/bin/env

# Set up and Activate Python 3.11 virtual environment
uv venv --seed ${HOME}/.local/bin/venv --python 3.11 --clear
source ${HOME}/.local/bin/venv/bin/activate
pip install --upgrade pip
```

#### XPK

Make sure you have the virtual environment activated when running XPK.

Install XPK and necessary tools:

```bash
# Install gcloud, if not already installed, https://cloud.google.com/sdk/docs/install
# Install kubectl, if not already installed, https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl

# Ensure to log in to your gcloud

# Install latest xpk
pip install xpk==1.8.0

# Install xpk pre-reqs kubectl-kueue and kjob (if you installed xpk via pip)
curl -LsSf https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/tags/v1.8.0/tools/install-xpk.sh -o install-xpk.sh
chmod +x install-xpk.sh
sudo ./install-xpk.sh
rm install-xpk.sh

# Follow https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin to install gke-gcloud-auth-plugin
```

#### Docker

Install Docker using instructions provided by your administrator. Once
installed, run the following commands:

```bash
## Configure docker and test installation
gcloud auth configure-docker
sudo usermod -aG docker $USER ## relaunch the terminal and make sure you have the virtual environment activated after running this command
docker run hello-world # Test docker
```

## Orchestration and deployment tools

For this recipe, the following setup is used:

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Pretraining job configuration and deployment** - XPK is used to configure
    and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the deepseek3-671b workload.

## Test environment

This recipe is optimized for and tested with tpu7x-4x8x8.

-   **GKE cluster** To create your GKE cluster, use the XPK instructions.
    [XPK instructions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#cluster-create).
    A sample command to create an XPK cluster is provided below.

### Environment Variables for Cluster Creation

The environment variables required for cluster creation and workload execution
are defined at the beginning of the `run_recipe.sh` script. **Before running the
`xpk workload create` command**, please open `run_recipe.sh` and modify the
`export` statements to set these variables to match your environment. It is
crucial to use consistent values for `PROJECT_ID`, `CLUSTER_NAME`, and `ZONE`
across all commands and configurations.

-   `PROJECT_ID`: Your GCP project name.
-   `CLUSTER_NAME`: The target cluster name.
-   `ZONE`: The zone for your cluster (e.g., `us-central1-c`).
-   `CONTAINER_REGISTRY`: The container registry to use (e.g., `gcr.io`).
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `MAXTEXT_ROOT`: The absolute path where you cloned the MaxText repository.
-   `WORKLOAD_IMAGE`: The Docker image for the workload. This is set in
    `run_recipe.sh` to
    `${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-deepseek-v3-runner` by
    default, matching the image built in the
    [Docker container image](#docker-container-image) section.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` using the following command:
    `export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-4096-fsdp")-$(date +%Y%m%d-%H%M)"`
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `ACCELERATOR_TYPE`: The TPU type (e.g., `tpu7x-4x4x4`). See topologies
    [here](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#configuration).
-   `RESERVATION_NAME`: Your TPU reservation name. Use the reservation name if
    within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

### Sample XPK Cluster Creation Command

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=${ACCELERATOR_TYPE} \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

## GCS Bucket setup
1. Create two buckets: one to hold the dataset and one to use for checkpoints. To create regional HNS buckets use the following commands:
```
# Set variables
export DATASET_BUCKET="dataloading-bucket-name"
export CHECKPOINT_BUCKET="checkpoint-bucket-name"
export REGION=""

# Create dataset bucket
gcloud storage buckets create gs://${DATASET_BUCKET} --location=${REGION}  --default-storage-class=Standard --enable-hierarchical-namespace --uniform-bucket-level-access

# Create checkpoint bucket  
gcloud storage buckets create gs://${CHECKPOINT_BUCKET} --location=${REGION}  --default-storage-class=Standard --enable-hierarchical-namespace --uniform-bucket-level-access
```
Replace the following values:  
- `<DATASET_BUCKET>`: the name of your Cloud Storage bucket with training dataset. Do not include the gs:// prefix  
- `<CHECKPOINT_BUCKET>`: the name of your Cloud Storage bucket where checkpoints will be written. Do not include the gs:// prefix
- `<REGION>`: the region where your GKE cluster is located ([available locations](https://cloud.google.com/storage/docs/locations#location-r))

2. Prepare your dataset in the DATASET_BUCKET. This recipe is configured to use the Grain loader with ArrayRecord files. Ensure your dataset files are accessible in this bucket. Follow these [instructions](https://github.com/AI-Hypercomputer/maxtext/blob/b93beba652db6b3f4e6c82dc48a83b03229f5d3a/getting_started/Data_Input_Pipeline.md#tfds-pipeline) to download the Allenai c4 dataset to the dataset bucket.
Then follow these [instructions](https://github.com/google/array_record/tree/main/beam) to convert the dataset into ArrayRecord.

3. GCSFuse lets you mount and access Cloud Storage buckets as local file systems, so applications can read and write objects in your bucket using standard file system semantics. You'll need to use the below commands to create [XPK storage resources](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#storage) for the dataset bucket in order to mount them to the MaxText workload using GCSFuse. For the dataset bucket use separate manifest file `dataset_pvc.yaml` from this repo.
Be sure to update `volumeHandle` in the yamls with your correct bucket names. Creating a bucket and attaching xpk storage is a one time setup.
```
# Set variables
export PROJECT=""
export CLUSTER=""
export ZONE=""

# Dataset Bucket PV/PVC
xpk storage attach dataset-gcsfuse-volume --type=gcsfuse --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mount-point=/tmp/dataset --readonly=false --bucket=$DATASET_BUCKET --size=64 --auto-mount=false --manifest=dataset_pvc.yaml
```

## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section below for installing
XPK and its dependencies. Docker installation is part of this process.

### Steps for building workload image

The following software versions are used:

-   Libtpu version: 0.0.35.dev20260121+nightly
-   Jax version: 0.8.1
-   Maxtext version: maxtext-tutorial-v1.1.0-1109-gcf051eb03
-   Python: 3.11
-   XPK: 1.8.0

Docker Image Building Command:

```bash
export CONTAINER_REGISTRY="" # Initialize with your registry
export CLOUD_IMAGE_NAME="${USER}-maxtext-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

# Set up and Activate Python 3.11 virtual environment for Docker build
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.11 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

# Make sure you're running on a Virtual Environment with python 3.11
if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" == "3.11" ]]; then { echo "You have the correct Python version 3.11"; } else { >&2 echo "Error: Python version must be 3.11"; false;} fi

# Clone MaxText Repository and Checkout Recipe Branch
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout maxtext-tutorial-v1.1.0-1109-gcf051eb03

# Build and upload the docker image
bash src/dependencies/scripts/docker_build_dependency_image.sh \
  MODE=nightly \
  JAX_VERSION=0.8.1 \
  LIBTPU_VERSION=0.0.35.dev20260121+nightly
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

# Deactivate the virtual environment
deactivate
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
running the benchmark, you can use the following gcloud command. (Note that XPK
does this for you already):

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

## Get the recipe
```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8/xpk
```

### Run deepseek3-671b Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the deepseek3-671b pretraining workload.

Before execution, use `nano ./run_recipe.sh` to edit the script and configure the environment variables to match your specific environment.

### Configuring and Starting workload

From the MaxText root directory, start your DeepSeek3-671B workload.

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the deepseek3-671b pretraining workload.

Edit the Recipe (run_recipe.sh) and populate the exported variables at the top of the file to match your environment.

```
# In run_recipe.sh, update these lines:
export PROJECT_ID="your-project-id"
export CLUSTER_NAME="your-cluster-name"
export ZONE="your-zone"
export BASE_OUTPUT_DIR="gs://<your_gcs_bucket>"
export DATASET_BUCKET_MOUNTED_PATH="/tmp/dataset" # Ensure this matches where XPK mounts the dataset bucket
```

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
-   **XLA Flags:** The `XLA_FLAGS` variable contains a set of XLA configurations
    optimized for this workload. These can be tuned for performance or
    debugging.
-   **MaxText Workload Overrides:** The `MAXTEXT_ARGS` variable holds the
    arguments passed to the `python3 -m src.MaxText.train` command. This
    includes model-specific settings like `per_device_batch_size`,
    `max_target_length`, and others. You can modify these to experiment with
    different model configurations.
-   **Virtual Environment:** The script activates the virtual environment
    created during the
    [Install XPK and dependencies](#install-xpk-and-dependencies) steps. If you
    used a different virtual environment, modify the `source` command at the top
    of `run_recipe.sh`.

Note that any MaxText configurations not explicitly overridden in `MAXTEXT_ARGS`
are expected to use the defaults within the specified `WORKLOAD_IMAGE`.

## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and stream logs:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods to find the specific name (e.g., deepseek3-0-0-xxxx)
kubectl get pods | grep ${WORKLOAD_NAME}
```
Then, stream the logs from the running pod (replace <POD_NAME> with the name you found):

```bash
kubectl logs -f <POD_NAME>
```
You can also monitor your cluster and TPU usage through the Google Cloud
Console.

### Follow Workload and View Metrics

After running `xpk workload create`, you will get a link to the Google Cloud
Console to view your workload logs. Example: `[XPK] Follow your workload here:
https://console.cloud.google.com/kubernetes/service/${ZONE}/${PROJECT_ID}/default/${WORKLOAD_NAME}/details?project=${PROJECT_ID}`
Alternatively, list workloads: (`xpk workload list`)

```bash
xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

For more in-depth debugging, use xpk inspector: (`xpk inspector`)

```bash
xpk inspector --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} [--workload ${WORKLOAD_NAME}]
```

### Delete resources

#### Delete a specific workload

```bash
xpk workload delete --workload ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
# Or filter and delete:
xpk workload delete --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} --filter-by-job=${USER}
```

#### Delete the entire XPK cluster

```bash
xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```

## Check results

After the job completes, you can check the results by:

-   Accessing output logs from your job.
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
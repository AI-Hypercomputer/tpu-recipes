# Inference Wan-AI/Wan2.1-T/I2V-14B-Diffusers, Wan-AI/Wan2.2-T/I2V-A14B-Diffusers. workload on Trillium GKE clusters with XPK.

This recipe outlines the steps for running a maxdiffusion
[Maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) pretraining workload on
[Trillium GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[XPK](https://github.com/AI-Hypercomputer/xpk).

## Workload Details

This workload is configured with the following details:

-   num_frames: 81
-   width: 1280
-   height: 720
-   Chips: 8/16 (2x4/4x4 topology)

## Prerequisites

To run this recipe, you need the following:

-   **GCP Project Setup:** Ensure you have a GCP project with billing enabled
    and have access to Trillium.
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
pip install xpk==1.3.0

# Install xpk pre-reqs kubectl-kueue and kjob (if you installed xpk via pip)
curl -LsSf https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/tags/v1.3.0/tools/install-xpk.sh -o install-xpk.sh
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
-   **Inference job configuration and deployment** - XPK is used to configure
    and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the Maxdiffusion Wan models.

## Test environment

This recipe is tested with tpu-v6e-8 and tpu-v6e-16.

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
-   `WORKLOAD_IMAGE`: The Docker image for the workload. This is set in
    `run_recipe.sh` to
    `${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-maxdiffusion-runner` by
    default, matching the image built in the
    [Docker container image](#docker-container-image) section.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` using the following command:
    `export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-Wan2_x")-$(date +%Y%m%d-%H%M)"`
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `ACCELERATOR_TYPE`: The TPU type (e.g., `v6e-8, v6e-16`). See topologies
    [here](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#configuration).
-   `RESERVATION_NAME`: Your TPU reservation name. Use the reservation name if
    within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

If you don't have a GCS bucket, create one with this command:

```bash
# Make sure BASE_OUTPUT_DIR is set in run_recipe.sh before running this.
gcloud storage buckets create ${BASE_OUTPUT_DIR} --project=${PROJECT_ID} --location=US  --default-storage-class=STANDARD --uniform-bucket-level-access
```

### Sample XPK Cluster Creation Command

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --device-type=${ACCELERATOR_TYPE} \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section below for installing
XPK and its dependencies. Docker installation is part of this process.

### Steps for building workload image

The following software versions are used:

-   Libtpu version: 0.0.37.dev20260224+nightly
-   Jax version: 0.9.0.1
-   Maxtext version: git+https://github.com/AI-Hypercomputer/maxdiffusion.git@85ba65e23c5fc04edcc923819f15bfa53ceb5a86
-   Python: 3.11
-   XPK: 1.3.0

Docker Image Building Command:

```bash
export CONTAINER_REGISTRY="" # Initialize with your registry
export CLOUD_IMAGE_NAME="${USER}-maxdiffusion-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"
export PROJECT_ID=<YOUR_PROJECT_ID>

# Clone MaxText Repository and Checkout Recipe Branch
git clone https://github.com/google/MaxDiffusion.git
cd MaxDiffusion

# Build and upload the docker image
bash docker_build_dependency_image.sh

# Connect to your project
gcloud config set project ${PROJECT_ID}

# Upload the image to your project's docker registry with the name ${CLOUD_IMAGE_NAME}
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
```

## Testing prompt

This recipe uses a single prompt for testing video generation speed

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
cd tpu-recipes/inference/inference/trillium/MaxDiffusion/inference/Wan2.x
```

### Run Maxdiffusion inference Workload

The `run_recipe_wan_2_x.sh` script contains all the necessary environment variables and
configurations to launch the Wan inference workload.

Before execution, use `nano ./run_recipe_wan_2_x.sh` to edit the script and configure the environment variables to match your specific environment.

To configure and run the benchmark:

```bash
# --- Environment Variables ---
export PROJECT_ID=<YOUR_PROJECT_ID>
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export ZONE=<YOUR_CLUSTER_ZONE>
export BASE_OUTPUT_DIR="" # E.g. gs://<YOUR_BUCKET_NAME>
export HF_TOKEN=<YOUR_HF_TOKEN>
export MODEL_NAME=<YOUR_MODEL_NAME> # Supported models: wan2-1-t2v, wan2-1-i2v, wan2_2-t2v, wan2_2-i2v
export TPU_TYPE=<YOUR_HARDWARE_TYPE> # v6e-8 or v6e-16 supported
export UV_VENV_PATH=<YOUR_VENV_PATH>
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

chmod +x run_recipe_wan_2_x.sh
nano ./run_recipe_wan_2_x.sh
./run_recipe_wan_2_x.sh ${MODEL_NAME}
```

You can customize the run by modifying `run_recipe_wan_2_x.sh`:

-   **T/I2V Models Support:** Both T2V and I2V models are supported. Follow the
    instructions inside run_recipe_wan_2_x.sh to switch between models.
-   **Environment Variables:** Adjust environmental variables like `PROJECT_ID`,
    `CLUSTER_NAME`, `ZONE`, `WORKLOAD_NAME`, `WORKLOAD_IMAGE`, and `BASE_OUTPUT_DIR`
    to match your environment.
-   **XLA Flags:** The `XLA_FLAGS` variable contains a set of XLA configurations
    optimized for this workload. These can be tuned for performance or
    debugging.
-   **MaxDiffusion Workload Overrides:** The `MAXDIFFUSION_ARGS` variable holds the
    arguments passed to the `python src/maxdiffusion/generate_wan.py` command. This
    includes model-specific settings like `per_device_batch_size`,
    `num_inference_steps`, and others. You can modify these to experiment with
    different model configurations.
-   **Virtual Environment:** The script activates the virtual environment
    created during the
    [Install XPK and dependencies](#install-xpk-and-dependencies) steps. If you
    used a different virtual environment, modify the `source` command at the top
    of `run_recipe_wan_2_x.sh`.

Note that any MaxDiffusion configurations not explicitly overridden in `MAXDIFFUSION_ARGS`
are expected to use the defaults within the specified `WORKLOAD_IMAGE`.


## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and stream logs:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods to find the specific name (e.g., ${WORKLOAD_NAME}-0-0-xxxx)
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

-   Video generated can be found in the Google Cloud Storage bucket specified by the
    `${BASE_OUTPUT_DIR}` variable in your `run_recipe.sh`.
-   Per video generation time (throughput) can be found by extracting the tensorboard content
    using event_accumulator inside tensorboard.backend.event_processing
-   Accessing output logs from your job.


## Next steps: deeper exploration and customization

This recipe is designed to provide a simple, reproducible "0-to-1" experience
for running a Maxdiffusion inference workload. Its primary purpose is to help you
verify your environment and achieve a first success with TPUs quickly and
reliably.


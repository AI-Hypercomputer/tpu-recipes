# Inference Wan-AI/Wan2.2-T2V-27B-Diffusers workload on Trillium GKE clusters with Cluster Toolkit

This recipe outlines the steps for running a maxdiffusion
[Maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) inference workload on
[Trillium GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).

## Workload Details

This workload is configured with the following details:

-   Model: Wan 2.2 Text-to-Video (T2V) 27B
-   num_frames: 81
-   width: 1280 (720p) or 832 (480p)
-   height: 720 (720p) or 480 (480p)
-   num_inference_steps: 40
-   fps: 16
-   TPU Cores: v6e-8, v6e-16

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
    in the [Install Cluster Toolkit and dependencies](#install-cluster-toolkit-and-dependencies) section
    to install Docker.
-   **Cluster Toolkit and Dependencies:** Follow the steps in the
    [Install Cluster Toolkit and dependencies](#install-cluster-toolkit-and-dependencies) section to
    install Cluster Toolkit (`gcluster`), `gcloud`, `kubectl`, and `gke-gcloud-auth-plugin`.

## Install Cluster Toolkit and dependencies

### Cluster Toolkit (gcluster)

Make sure you have Cluster Toolkit (`gcluster`) added to your `PATH`.

Install Cluster Toolkit (`gcluster`) and necessary tools:

```bash
# Install gcloud, if not already installed, https://cloud.google.com/sdk/docs/install
# Install kubectl, if not already installed, https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl

# Ensure to log in to your gcloud

# Install Cluster Toolkit (gcluster)
# Download and install Cluster Toolkit (gcluster) v1.104.0
curl -L -O "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/v1.104.0/gcluster_bundle_linux_amd64.tgz"
mkdir -p "${HOME}/cluster-toolkit" && tar -xzf gcluster_bundle_linux_amd64.tgz -C "${HOME}/cluster-toolkit" && rm gcluster_bundle_linux_amd64.tgz
export PATH="${HOME}/cluster-toolkit:${PATH}"

# Follow https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin to install gke-gcloud-auth-plugin
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
-   **Inference job configuration and deployment** - Cluster Toolkit (`gcluster`) is used to configure
    and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the Maxdiffusion Wan models.

## Test environment

This recipe is tested with `v6e-8` and `v6e-16`.

-   **GKE cluster** To create your GKE cluster, use the [Cluster Toolkit Cloud TPU deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview).
    A sample command to create a Cluster Toolkit cluster is provided below.

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
-   `CONTAINER_REGISTRY`: The container registry to use (e.g., `gcr.io`).
-   `BASE_OUTPUT_DIR`: Output directory for model logs/artifacts (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `WORKLOAD_IMAGE`: The Docker image for the workload. This is set to a placeholder
    `<YOUR_CONTAINER_REGISTRY>/<YOUR_PROJECT_ID>/<YOUR_IMAGE_NAME>:latest` by default.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` using the following command:
    `export WORKLOAD_NAME="$(printf "%.14s" "${USER//_/-}-wan2-2-t2v")-$(date +%Y%m%d-%H%M)"`
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `ACCELERATOR_TYPE`: The TPU type (e.g., `v6e-8` or `v6e-16`). See topologies
    [here](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#configuration).
-   `RESERVATION_NAME`: Your TPU reservation name. Use the reservation name if
    within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

If you don't have a GCS bucket, create one with this command:

```bash
# Make sure BASE_OUTPUT_DIR is set in run_recipe.sh before running this.
gcloud storage buckets create ${BASE_OUTPUT_DIR} --project=${PROJECT_ID} --location=US  --default-storage-class=STANDARD --uniform-bucket-level-access
```

### Sample Cluster Toolkit Cluster Creation Command

```bash
gcluster deploy examples/gke-tpu-v6e/gke-tpu-v6e.yaml \
  --backend-config="bucket=${PROJECT_ID}-ctk-tf-state" \
  --vars="project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME},region=${ZONE%-*},zone=${ZONE},num_slices=1,reservation=${RESERVATION_NAME}"
```

## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section below for installing
Cluster Toolkit and its dependencies. Docker installation is part of this process.

### Steps for building workload image

The following software versions are used:

-   Libtpu version: 0.0.40 or nightly
-   Jax version: 0.10.0 or nightly
-   MaxDiffusion version: git+https://github.com/AI-Hypercomputer/maxdiffusion.git
-   Python: 3.12
-   Cluster Toolkit: 1.104.0

Docker Image Building Command:

```bash
export PROJECT_ID=<YOUR_PROJECT_ID>
export CONTAINER_REGISTRY="" # Initialize with your registry
export CLOUD_IMAGE_NAME="${USER}-maxdiffusion-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

# Clone MaxDiffusion Repository
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion

# Build and upload the docker image
bash docker_build_dependency_image.sh

# Connect to your project
gcloud config set project ${PROJECT_ID}

# Upload the image to your project's docker registry with the name ${CLOUD_IMAGE_NAME}
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
```

## Testing prompt

This recipe uses a single prompt for testing video generation speed.

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
cd tpu-recipes/inference/trillium/MaxDiffusion/Wan2.x/Wan2.2-T2V/cluster_toolkit
```

### Run Maxdiffusion inference Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the Wan inference workload.

Before execution, use `nano ./run_recipe.sh` to edit the script and configure the environment variables to match your specific environment.

To configure and run the benchmark:

```bash
# --- Environment Variables ---
export PROJECT_ID=<YOUR_PROJECT_ID>
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export ZONE=<YOUR_CLUSTER_ZONE>
export BASE_OUTPUT_DIR="" # E.g. gs://<YOUR_BUCKET_NAME>
export HF_TOKEN=<YOUR_HF_TOKEN>
export TPU_TYPE=<YOUR_HARDWARE_TYPE> # Supported values: v6e-8, v6e-16
export RESOLUTION=<720p or 480p> # Supported: 720p, 480p (Defaults to 720p)
export WORKLOAD_IMAGE=<YOUR_WORKLOAD_IMAGE> # E.g. gcr.io/<YOUR_PROJECT_ID>/<YOUR_IMAGE_NAME> or nightly pre-built image

chmod +x run_recipe.sh
nano ./run_recipe.sh
./run_recipe.sh
```

You can customize the run by modifying `run_recipe.sh`:

-   **Environment Variables:** Adjust environmental variables like `PROJECT_ID`,
    `CLUSTER_NAME`, `ZONE`, `WORKLOAD_NAME`, `WORKLOAD_IMAGE`, and `BASE_OUTPUT_DIR`
    to match your environment.
-   **XLA Flags:** The `XLA_FLAGS` variable contains a set of XLA configurations
    optimized for Trillium TPUs. These can be tuned for performance or
    debugging.
-   **MaxDiffusion Workload Overrides:** The `MAXDIFFUSION_ARGS` variable holds the
    arguments passed to the `python src/maxdiffusion/generate_wan.py` command. This
    includes model-specific settings like `per_device_batch_size`,
    `num_inference_steps`, and others. You can modify these to experiment with
    different model configurations.

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

After running `gcluster job submit`, you will get a link to the Google Cloud
Console to view your workload logs. Example: `Follow your workload here:
https://console.cloud.google.com/kubernetes/service/${ZONE}/${PROJECT_ID}/default/${WORKLOAD_NAME}/details?project=${PROJECT_ID}`
Alternatively, list workloads: (`gcluster job list`)

```bash
gcluster job list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

For more in-depth debugging, inspect the job: (`gcluster job inspect`)

```bash
gcluster job inspect --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE} --name ${WORKLOAD_NAME}
```

### Delete resources

#### Delete a specific workload

```bash
gcluster job cancel ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

#### Delete the entire Cluster Toolkit cluster

```bash
gcluster destroy ${CLUSTER_NAME} --auto-approve
```

## Check results

After the job completes, you can check the results by:

-   Video generated can be found in the Google Cloud Storage bucket specified by the
    `${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}` variable.
-   Per video generation time (throughput) can be found by extracting the tensorboard content
    using event_accumulator inside tensorboard.backend.event_processing.
-   Accessing output logs from your job.


## Next steps: deeper exploration and customization

This recipe is designed to provide a simple, reproducible "0-to-1" experience
for running a Maxdiffusion inference workload on Trillium. Its primary purpose is to help you
verify your environment and achieve a first success with TPUs quickly and
reliably.

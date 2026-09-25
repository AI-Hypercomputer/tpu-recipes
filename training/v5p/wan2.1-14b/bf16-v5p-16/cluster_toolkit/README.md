# Pretrain wan workload on v5p GKE clusters with Cluster Toolkit

<!-- disableFinding(LINK_ID) -->

This recipe outlines the steps for running a wan
[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) pretraining
workload on
[Cloud TPU v5p GKE clusters](https://cloud.google.com/kubernetes-engine)
by using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).

This is a v5p port of the Ironwood (tpu7x) wan2.1-14b recipe, targeting a single
`v5p-16` slice.

## Prerequisites

<!-- disableFinding(LINK_ID) -->

To run this recipe, you need the following:

-   **GCP Project Setup:** Ensure you have a GCP project with billing enabled
    and a TPU v5p reservation (or quota) available.
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
-   **Pretraining job configuration and deployment** - Cluster Toolkit (`gcluster`) is used to configure
    and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the wan workload.


## Test environment

<!-- disableFinding(LINK_ID) -->

This recipe is optimized for and tested with a single `v5p-16` slice.

-   **Topology:** `v5p-16` is a single slice of 8 chips arranged in a `2x2x2`
    topology across 2 hosts.
-   **Devices:** v5p uses megacore, exposing **1 JAX device per chip**, so
    `v5p-16` = **8 devices**.
-   **Collectives:** All 8 chips are on the same slice, so every collective runs
    over **ICI (no DCN / cross-slice traffic)**. Unlike the 2x `v5p-8`
    multislice variant, there is no cross-slice DCN backpressure here, so
    ICI-only collectives should yield higher MFU.
-   **Mesh (single slice, ICI only):**
    `ici_data=1 * ici_fsdp=8 * ici_tensor=1 = 8 devices`. There is no `dcn_*`
    parallelism.

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
-   `ZONE`: The zone for your cluster (e.g., `us-central1-a`).
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `DATASET_DIR`: Location of the preprocessed TFRecord dataset (defaults to
    `${BASE_OUTPUT_DIR}/wan_tfr_dataset_pusa_v1`).
-   `WORKLOAD_IMAGE`: The Docker image for the workload (e.g.,
    `gcr.io/${PROJECT_ID}/${USER}-maxdiffusion-runner:latest`), matching the
    image built in the [Docker container image](#docker-container-image)
    section. This image must be pushed before running the workload.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` to `$(printf "%.14s" "${USER//_/-}-wan21")-$(date +%Y%m%d-%H%M)`
    by default.
-   `DEVICE_TYPE`: The TPU device type, `v5p-16`.
-   `RESERVATION_NAME`: Your TPU v5p reservation name. Use the reservation name
    if within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

If you don't have a GCS bucket, create one with this command:

```bash
# Make sure BASE_OUTPUT_DIR is set in run_recipe.sh before running this.
gcloud storage buckets create ${BASE_OUTPUT_DIR} --project=${PROJECT_ID} --location=US  --default-storage-class=STANDARD --uniform-bucket-level-access
```

### Sample Cluster Toolkit Cluster Creation Command

A `v5p-16` slice requires a multi-host `ct5p-hightpu-4t x2` (`2x2x2`) node pool
in the cluster.

```bash
gcluster deploy examples/gke-tpu-v5p/gke-tpu-v5p.yaml \
  --backend-config="bucket=${PROJECT_ID}-ctk-tf-state" \
  --vars="project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME},region=${ZONE%-*},zone=${ZONE},num_slices=1,reservation=${RESERVATION_NAME}"
```


## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section below for installing
Cluster Toolkit and its dependencies. Docker installation is part of this process.

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

# Change to your home directory and clone maxdiffusion
cd ~/
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion
git checkout v3

# Run WAN 2.1 Docker build
export RECIPE_DOCKER_IMAGE=maxdiffusion_base_image
bash docker_build_dependency_image.sh mode=nightly
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
```

## Training dataset

Before training, prepare the video training dataset. For this example, we will
be using the PusaV1 dataset.

```bash
cd ~/maxdiffusion
# Set to TPU or CPU based on your local setup
# bash setup.sh MODE=stable DEVICE=cpu
bash setup.sh MODE=stable DEVICE=tpu

# Following assumes that you have mounted an external drive to your VM
# and you have created a mount point for `/mnt/disks/external_disk`
# along with given write permissions to the directories described below.
export HF_TOKEN=<token>
export HF_DATASET_DIR=/mnt/disks/external_disk/PusaV1_training/
export TFRECORDS_DATASET_DIR=/mnt/disks/external_disk/wan_tfr_dataset_pusa_v1
export HF_HUB_CACHE=/mnt/disks/external_disk/maxdiffusion_hf_cache/
export DATASET_DIR=${BASE_OUTPUT_DIR}/wan_tfr_dataset_pusa_v1

# Download the dataset
huggingface-cli download RaphaelLiu/PusaV1_training --repo-type dataset --local-dir ${HF_DATASET_DIR}

# Preprocess and convert the dataset to TFRecord format
# This can be done outside of the GKE cluster, or a TPU machine. We recommend using a CPU machine for this preprocessing step. Use skip_jax_distributed_system=True when running on cpu machine.
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
running the benchmark, you can use the following gcloud command.:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

## Get the recipe
```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/v5p/wan2.1-14b/bf16-v5p-16/cluster_toolkit
```

### Run wan Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the wan pretraining workload.

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
    settings and the mesh configuration
    (`ici_data_parallelism=1 ici_fsdp_parallelism=8 ici_tensor_parallelism=1`).

Note that any MaxDiffusion configurations not explicitly overridden in
`MAXDIFFUSION_ARGS` are expected to use the defaults within the specified
`WORKLOAD_IMAGE`.

## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and stream logs:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods to find the specific name (e.g., <workload_name>-slice-job-0-0-xxxx)
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
https://console.cloud.google.com/kubernetes/service/${ZONE}/${PROJECT_ID}/default/<workload_name>/details?project=${PROJECT_ID}`
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
gcluster job cancel <workload_name> --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

#### Delete the entire Cluster Toolkit cluster

```bash
gcluster destroy ${CLUSTER_NAME} --auto-approve
```


## Check results

After the job completes, you can check the results by:

-   Accessing output logs from your job.
-   Checking any data stored in the Google Cloud Storage bucket specified by the
    `${BASE_OUTPUT_DIR}` variable in your `run_recipe.sh`.
-   Reviewing metrics in Cloud Monitoring, if configured.


## Next steps: deeper exploration and customization

This recipe provides a starting point for running MaxDiffusion workloads. For
advanced usage, including exploring different models, datasets, and training
parameters, please refer to the
[MaxDiffusion GitHub repository](https://github.com/AI-Hypercomputer/maxdiffusion).

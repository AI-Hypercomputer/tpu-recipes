# Pretrain llama3-1-70b workload on Ironwood GKE clusters with XPK

This recipe outlines the steps for running a llama3-1-70b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[XPK](https://github.com/AI-Hypercomputer/xpk).


## Workload Details

This workload is configured with the following details:

-   Sequence Length: 8192
-   Precision: bfloat16
-   Chips: 64 (4x4x4 topology)

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
pip install xpk==0.16.0

# Install xpk pre-reqs kubectl-kueue and kjob (if you installed xpk via pip)

# Download kueuectl - https://kueue.sigs.k8s.io/docs/reference/kubectl-kueue/installation/#installing-from-release-binaries
curl -Lo ./kubectl-kueue https://github.com/kubernetes-sigs/kueue/releases/download/v0.12.2/kubectl-kueue-linux-amd64

# Make binary executable
chmod +x ./kubectl-kueue

# Move to location in system PATH
sudo mv ./kubectl-kueue /usr/local/bin/kubectl-kueue

# Download kjob - https://github.com/kubernetes-sigs/kjob/blob/main/docs/installation.md
curl -Lo ./kubectl-kjob https://github.com/kubernetes-sigs/kjob/releases/download/v0.1.0/kubectl-kjob-linux-amd64

# Make binary executable
chmod +x ./kubectl-kjob

# Move to location in system PATH
sudo mv ./kubectl-kjob /usr/local/bin/kubectl-kjob

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
    resource, which manages the execution of the MaxText pretraining workload.

## Test environment

This recipe is optimized for and tested with tpu7x-4x4x4.

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
    `${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-maxtext-runner` by default,
    matching the image built in the
    [Docker container image](#docker-container-image) section.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` using the following command:
    `export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-llama3-1-70b")-\
    $(date +%Y%m%d-%H%M)"`
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `ACCELERATOR_TYPE`: The TPU type (e.g., `tpu7x-4x4x4`). See topologies
    [here](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#configuration).
-   `RESERVATION_NAME`: Your TPU reservation name. Use the reservation name if
    within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

If you don’t have a GCS bucket, create one with this command:

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
  --tpu-type=${ACCELERATOR_TYPE} \
  --num-slices=1 \
  --on-demand \
  --reservation=${RESERVATION_NAME}
```

## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section below for installing
XPK and its dependencies. Docker installation is part of this process.

### Steps for building workload image

**Warning:** If any of the software versions below show as "N/A", you *must*
fill in the correct versions. To find the missing versions (e.g., for MaxText
commit hash, Libtpu, and Jax/Jaxlib), you may need to:
1.  Pull the Docker image from the workload that this recipe is based on.
2.  Start the Docker container.
3.  Run commands within the container to get the specific versions. For example,
to find the MaxText commit, you can use `git rev-parse HEAD` inside the cloned
MaxText repository within the container. For Python package versions, use
`pip show <package_name>`.

The following software versions are used:

-   Libtpu version: N/A
-   Jax version: N/A
-   Maxtext version: N/A
-   Python 3.11
-   XPK 0.16.0

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
[[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" == "3.12" ]] || { >&2 echo "Error: Python version must be 3.12."; false; }

# Clone MaxText Repository and Checkout Recipe Branch
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout N/A

# Custom Jax and LibTPU wheels
pip download libtpu==N/A -f"https://storage.googleapis.com/jax-releases/libtpu_releases.html"

pip download --pre jax==N/A jaxlib==N/A --index https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

# Build and upload the docker image
bash dependencies/scripts/docker_build_dependency_image.sh MODE=custom_wheels
bash dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

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

### Run llama3-1-70b Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the llama3-1-70b pretraining workload.

To run the benchmark, first make the script executable and then run it:

```bash
chmod +x run_recipe.sh
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
and logs:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}
kubectl logs -f -n default jobset/${WORKLOAD_NAME}-0-worker-0
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
xpk inspector --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} [--workload <workload_name>]
```

### Delete resources

#### Delete a specific workload

```bash
xpk workload delete --workload <workload_name> --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
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

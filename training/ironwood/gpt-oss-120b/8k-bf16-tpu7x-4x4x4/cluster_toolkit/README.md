# Pretrain gpt-oss-120b workload on Ironwood GKE clusters with Cluster Toolkit

This recipe outlines the steps for running a gpt-oss-120b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).


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
-   **Docker (Optional):** Only required if building a custom workload image from source instead of using the pre-built `WORKLOAD_IMAGE` default (`gcluster` itself does not require Docker). Follow the steps
    in the [Install Cluster Toolkit and dependencies](#install-cluster-toolkit-and-dependencies) section
    to install Docker if needed.
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

### Docker (Optional — only required if building a custom image from source)

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
    resource, which manages the execution of the gpt-oss-120b workload.


## Test environment

This recipe is optimized for and tested with tpu7x-4x4x4.

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
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `WORKLOAD_IMAGE`: The path to your custom MaxText Docker image (e.g.,
    `"${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-maxtext-runner"`).
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` using the following command:
    `export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.10s" "${USER//_/-}")-gptoss120b-$(date +%H%M)}"`
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `ACCELERATOR_TYPE`: The TPU type (e.g., `tpu7x-4x4x4`). See topologies
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
gcluster deploy examples/gke-tpu-7x/gke-tpu-7x.yaml \
  --backend-config="bucket=${PROJECT_ID}-ctk-tf-state" \
  --vars="project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME},region=${ZONE%-*},zone=${ZONE},num_slices=1,reservation=${RESERVATION_NAME}"
```


## Docker container image (Optional — only required if building a custom image from source)

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section below for installing
Cluster Toolkit and its dependencies. Docker installation is part of this process.

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

-   Libtpu version: 0.0.47
-   Jax version: 0.11.2.dev20260914
-   Maxtext version: 4e2396c
-   Python: 3.12
-   Cluster Toolkit: 1.104.0

Docker Image Building Command:

```bash
export CONTAINER_REGISTRY="" # Initialize with your registry
export CLOUD_IMAGE_NAME="${USER}-maxtext-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

# Clone MaxText Repository and Checkout Recipe Branch
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout 4e2396c

# Build and upload the docker image
bash src/dependencies/scripts/docker_build_dependency_image.sh \
  MODE=nightly \
  JAX_VERSION=0.11.2.dev20260914 \
  LIBTPU_VERSION=0.0.47
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

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

### Run gpt-oss-120b Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the gpt-oss-120b pretraining workload.

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

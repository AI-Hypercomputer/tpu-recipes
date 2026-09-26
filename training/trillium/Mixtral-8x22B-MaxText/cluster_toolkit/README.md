# Pretrain Mixtral-8x22B workload on Trillium GKE clusters with Cluster Toolkit

This recipe outlines the steps for running a Mixtral-8x22B
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Trillium GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).


## Workload Details

This workload is configured with the following details:

-   Sequence Length: 4096
-   Precision: bfloat16
-   Chips: 256 per slice (v6e-256, 16x16 topology, 64 hosts per slice);
    `NUM_SLICES` in `run_recipe.sh` defaults to 1 and the recipe was
    published for 1, 10, 20, 30 or 40 slices

The MaxText arguments and XLA flags in `run_recipe.sh` are the `mixtral_8x22b_dropped`
model configuration from
[maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/tpu-recipes-v0.1.2/benchmarks/maxtext_trillium_model_configs.py)
at MaxText `tpu-recipes-v0.1.2` (profiler settings and the unused
`dataset_path` omitted), expanded inline so the script does not depend on
the MaxText `benchmark_runner`.

## Prerequisites

To run this recipe, you need the following:

-   **GCP Project Setup:** Ensure you have a GCP project with billing enabled
    and quota for Trillium (TPU v6e).
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
    resource, which manages the execution of the Mixtral-8x22B workload.


## Test environment

This recipe is optimized for v6e-256 slices (16x16 topology, 64 `ct6e-standard-4t` hosts per slice).

-   **GKE cluster** To create your GKE cluster, refer to the
    [Cloud TPU deployments overview](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview)
    and the Cluster Toolkit
    [Trillium (v6e) GKE blueprint](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples/gke-tpu-v6e).
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
-   `ZONE`: The zone for your cluster (e.g., `us-east5-b`).
-   `REGION`: The region for your cluster (e.g., `us-east5`). Can be derived as `${ZONE%-*}`.
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `WORKLOAD_IMAGE`: The Docker image for the workload, i.e. the image built
    in the [Docker container image](#docker-container-image) section.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` to `${USER}-mixtral-8x22b-$(date +%H%M)` by default.
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
clusters with Cloud TPU node pools. For detailed deployment instructions and
configuration options for Trillium (v6e), refer to the
[gke-tpu-v6e example README](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples/gke-tpu-v6e).

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

Fill in `examples/gke-tpu-v6e/gke-tpu-v6e-deployment.yaml` with your project,
region, zone, the number of slices you plan to run (`NUM_SLICES`), `machine_type: ct6e-standard-4t`,
`tpu_topology: 16x16` and your reservation (see the example README), then deploy
it with `gcluster deploy`:

```bash
cd ~/cluster-toolkit
./gcluster deploy -d examples/gke-tpu-v6e/gke-tpu-v6e-deployment.yaml \
  examples/gke-tpu-v6e/gke-tpu-v6e.yaml
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

To build your own image, follow the steps in this section. If you don't have
Docker installed on your workstation, see the section above for installing
Cluster Toolkit and its dependencies. Docker installation is part of this
process.

### Steps for building workload image

The following software versions are used:

-   Maxtext version: `tpu-recipes-v0.1.2`
-   Base image: `us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1`
    (JAX 0.5.2)
-   Cluster Toolkit: 1.104.0

Docker Image Building Command:

```bash
export CLOUD_IMAGE_NAME="${USER}_runner"

# Clone MaxText Repository and Checkout Recipe Tag
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout tpu-recipes-v0.1.2

# Build and upload the docker image (pushed to gcr.io/<gcloud project>/${CLOUD_IMAGE_NAME}:latest)
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

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

### Run Mixtral-8x22B Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the Mixtral-8x22B pretraining workload.

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
-   **Number of slices:** `NUM_SLICES` (default `1`) sets `--num-slices`.
    The recipe was published for 1, 10, 20, 30 or 40 v6e-256 slices. Your cluster
    needs that many v6e-256 node pools.
-   **XLA Flags:** The `XLA_FLAGS` variable contains a set of XLA configurations
    optimized for this workload. These can be tuned for performance or
    debugging.
-   **MaxText Workload Overrides:** The `MAXTEXT_ARGS` variable holds the
    arguments passed to the `python3 -m MaxText.train` command. This includes
    model-specific settings like `per_device_batch_size`, `max_target_length`,
    and others. You can modify these to experiment with different model
    configurations.

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

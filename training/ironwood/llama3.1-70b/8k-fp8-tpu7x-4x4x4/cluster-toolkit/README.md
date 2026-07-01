# Pretrain llama3-1-70b workload on Ironwood GKE clusters with Cluster Toolkit

This recipe outlines the steps for running a llama3-1-70b [MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on [Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)'s `gcluster` command-line interface.

## Workload Details

- **Sequence Length:** 8192
- **Precision:** fp8 (Full Quantization)
- **Chips:** 64 (4x4x4 TPU v7x/Ironwood topology)

---

## Prerequisites

To run this recipe, you need:

1. **GCP Project Setup:** Ensure you are using standard projects containing TPU v7x clusters (e.g., `<YOUR_PROJECT_ID>`).
2. **User Project Permissions:** The executing account requires standard IAM Roles:
   * Compute Admin
   * Kubernetes Engine Admin
   * Storage Admin
   * Service Account User
   * Logging/Monitoring Admin
3. **Docker:** Installed and configured with GCP credentials:
   ```bash
   gcloud auth configure-docker
   ```
4. **Cluster Toolkit CLI (gcluster):** Downloaded and installed.

---

## Install Cluster Toolkit (gcluster)

Unlike XPK, the Cluster Toolkit operates as a single pre-compiled Go binary. No Python virtual environment or pip dependencies are required to run `gcluster`.

### Install gcluster Binary

Run the following commands on your workstation to download and extract the recommended release bundle (v1.89.0 or later):

```bash
# Define target version, OS, and Architecture
TAG="v1.89.0"
OS="linux"
ARCH="amd64"

# Create installation directory, download and extract the platform-specific bundle
mkdir -p ~/cluster-toolkit
curl -L https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/${TAG}/gcluster_bundle_${OS}_${ARCH}.zip -o ~/cluster-toolkit/gcluster_bundle.zip
unzip ~/cluster-toolkit/gcluster_bundle.zip -d ~/cluster-toolkit/

# Add to PATH for easy command access
export PATH=$PATH:${HOME}/cluster-toolkit

# Verify installation
gcluster --version
```

---

## Connecting to an Existing GKE TPU v7x Cluster

This recipe targets active **tpu7x-4x4x4** clusters running on your target GCP projects.

1. Authenticate and log in:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>
   ```

2. Set target project context and list active GKE clusters:
   ```bash
   # Using your target GCP project ID as an example
   export PROJECT_ID="<YOUR_PROJECT_ID>"
   gcloud container clusters list --project=${PROJECT_ID}
   ```

3. Fetch credentials and connect `kubectl` to your cluster:
   ```bash
   # Replace CLUSTER_NAME and ZONE with the target v7x cluster details
   export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
   export ZONE="<YOUR_CLUSTER_ZONE>" # e.g., us-central1-c
   
   gcloud container clusters get-credentials ${CLUSTER_NAME} \
     --project=${PROJECT_ID} \
     --zone=${ZONE}
   ```

4. Verify the nodes and TPU selectors are active:
   ```bash
   kubectl get nodes -o wide
   kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x --show-labels
   ```

---

## Docker Container Image

To build the workload container image optimized for MaxText pretraining:

```bash
# Set environment variables
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CONTAINER_REGISTRY="us-docker.pkg.dev"  # e.g., Artifact Registry
export CLOUD_IMAGE_NAME="maxtext-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/gcluster-repo/${CLOUD_IMAGE_NAME}:latest"

# Clone MaxText and checkout the required tag/commit
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout a0fceb5

# Build the container image
bash src/dependencies/scripts/docker_build_dependency_image.sh \
  MODE=nightly \
  JAX_VERSION=0.9.2.dev20260306 \
  LIBTPU_VERSION=0.0.37

# Tag and push the built image to your Artifact Registry
docker tag maxtext_nightly ${WORKLOAD_IMAGE}
docker push ${WORKLOAD_IMAGE}
```

---

## Run the Recipe

### 1. Configure environment settings
Open the [run_recipe.sh](run_recipe.sh) script and configure the required GCP variables:
* `PROJECT_ID` (Your GCP Project ID)
* `CLUSTER_NAME` (Your existing cluster name)
* `ZONE` (Your existing cluster zone/location)
* `BASE_OUTPUT_DIR` (Cloud Storage bucket for MaxText checkpoints/outputs)
* `ARTIFACT_DIR` (Cloud Storage bucket for log archives)
* `WORKLOAD_IMAGE` (Your pushed Docker image)

### 2. Submit Workload Job
Submit the pretraining job:
```bash
chmod +x run_recipe.sh
./run_recipe.sh
```

Under the hood, this compiles and submits a highly optimized Kubernetes JobSet manifest that maps the resources to your Kueue queues.

---

## Monitor the Workload

### 1. List Active Jobs
Use `gcluster` to monitor the status of jobs running on the cluster:
```bash
gcluster job list --project=${PROJECT_ID} --cluster=${CLUSTER_NAME} --location=${ZONE}
```

### 2. View Live Output Logs
Fetch and follow output logs from worker-0 in real-time:
```bash
gcluster job logs <WORKLOAD_NAME> -f --project=${PROJECT_ID} --cluster=${CLUSTER_NAME} --location=${ZONE}
```

Alternatively, monitor the status directly using standard `kubectl` commands:
```bash
# Inspect pods in the default namespace
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=<WORKLOAD_NAME>

# Tail logs
kubectl logs -f jobset/<WORKLOAD_NAME>-0-worker-0
```

---

## Cleanup

To avoid incurring unnecessary cloud costs:

### 1. Cancel Job
```bash
gcluster job cancel <WORKLOAD_NAME> --project=${PROJECT_ID} --cluster=${CLUSTER_NAME} --location=${ZONE}
```

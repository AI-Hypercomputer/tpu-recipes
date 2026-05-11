# Instructions to run qwen3-30b-a3b RL on Ironwood GKE Cluster with XPK

This recipe outlines the steps for running Reinforcement Learning (RL) on the `qwen3-30b-a3b` model using MaxText on Ironwood GKE cluster.

## Prerequisites
To run this recipe, you need the following:

* A read-access token from [HuggingFace](https://huggingface.co/settings/tokens) is required to access the `qwen3-30b-a3b` model.
* Ensure you have a GCP project with billing enabled and are allowlisted for Ironwood access.
* Ensure your user account or service account has the following IAM permissions:
  * Artifact Registry Writer
  * Compute Admin
  * Kubernetes Engine Admin
  * Logging Admin
  * Monitoring Admin
  * Service Account User
  * Storage Admin
  * Vertex AI Administrator
  * Service Usage Consumer
  * TPU Viewer
* Docker must be installed and configured for sudoless use on your workstation. Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).
* Follow the steps in the [XPK Prerequisites section](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md#1-prerequisites) to install prerequisites for XPK.
* Ensure that you have an Ironwood `tpu7x-128` GKE pathways cluster set up and that you have the necessary permissions to deploy workloads to it. If you don't have a cluster set up, follow the instructions in the [documentation](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster) to create one.
* Ensure Python 3.12 is installed on your workstation.

## Create Virtual Python Environment
Run the following to create a virtual Python environment:

### Setup uv

```bash
sudo apt update
curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh
chmod +x install-uv.sh
./install-uv.sh
rm install-uv.sh
source ${HOME}/.local/bin/env
```

### Setup and Activate Python 3.12 Virtual Environment

```bash
uv venv --seed ${HOME}/.local/bin/venv --python 3.12 --clear
source ${HOME}/.local/bin/venv/bin/activate
pip install --upgrade pip
```

## Build MaxText Docker Image

### Test Docker Installation

```bash
# Authenticate your user account for gcloud CLI access
gcloud auth login

# Configure application default credentials for Docker
gcloud auth application-default login

# Configure Docker credentials and test your access
gcloud auth configure-docker
docker run hello-world
```

### Install MaxText

```bash
uv pip install maxtext[runner]==0.2.2 --resolution=lowest
```

### Build MaxText Docker Image

```bash
build_maxtext_docker_image WORKFLOW=post-training
```

### Upload MaxText Docker Image to Artifact Registry

```bash
export PROJECT_ID="" # Set this to your GCP project ID where the Ironwood cluster is deployed
export CLOUD_IMAGE_NAME="${USER}-maxtext-rl"

gcloud config set project ${PROJECT_ID}
upload_maxtext_docker_image CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
```

### Deactivate Virtual Environment
```bash
deactivate
```

## Run qwen3-30b-a3b RL Recipe

```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/qwen3-30b-a3b/tpu7x-4x4x4/rl
```

The `run_recipe.sh` script is the main entry point for launching the `qwen3-30b-a3b` RL workload. It contains all the necessary environment variables and configurations.

Before execution, use `nano ./run_recipe.sh` to edit the script and configure the environment variables to match your specific environment.

To launch the workload:

```bash
chmod +x run_recipe.sh
nano ./run_recipe.sh
./run_recipe.sh
```

You can customize the run by modifying `run_recipe.sh`:

* Environment Variables: Variables like `PROJECT_ID`, `CLUSTER_NAME`, `ZONE`, `BASE_OUTPUT_DIR`, `MAXTEXT_CKPT_PATH`, and `HF_TOKEN` are defined at the beginning of the script. Adjust these to match your environment.
* XLA Flags: The `XLA_FLAGS` variable contains a set of XLA configurations optimized for this workload. These can be tuned for performance or debugging.
* Virtual Environment: The script activates the virtual environment created during the `Setup and Activate Python 3.12 Virtual Environment` step. If you used a different virtual environment, modify the source command at the top of `run_recipe.sh`.

## Monitor the Workload
To monitor your job's progress, you can use `kubectl` to check the `Jobset` status and stream logs directly from the pods.

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods to find the specific name
kubectl get pods | grep ${WORKLOAD_NAME}

# stream the logs from the running pod (replace <POD_NAME> with the name you found)
kubectl logs -f <POD_NAME>
```

Alternatively, after running `xpk workload create-pathways` command, you will get a link to the Google Cloud Console to view your workload logs. Example: 

```bash
Follow your workload here: https://console.cloud.google.com/kubernetes/service/${ZONE}/${PROJECT_ID}/default/${WORKLOAD_NAME}/details?project=${PROJECT_ID} 
```
Follow the link to view logs and monitor your workload's progress in the Cloud Console.

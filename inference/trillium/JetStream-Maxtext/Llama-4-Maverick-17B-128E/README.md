# Inference benchmark of Llama-4-Maverick-17B-128E with JetStream, MaxText and Pathways on Cloud with TPU v6e (Trillium)

This recipe outlines the steps to benchmark the inference of a Llama-4-Maverick-17B-128E model using [JetStream](https://github.com/AI-Hypercomputer/JetStream/tree/main) with [MaxText](https://github.com/AI-Hypercomputer/maxtext) engine on multihost [TPU v6e Node slices](https://cloud.google.com/kubernetes-engine) utilizing Pathways on Cloud.

* [JetStream](https://github.com/AI-Hypercomputer/JetStream) is a throughput and memory-optimized engine for LLM inference on XLA devices, primarily TPUs written in JAX.
* [MaxText](https://github.com/AI-Hypercomputer/maxtext) is an open-source LLM project by Google, written in JAX and designed to be highly performant and scalable, running efficiently on Google Cloud TPUs and GPUs. 
* [Pathways](https://github.com/google/pathways-job) is a system that simplifies large-scale ML computations by enabling a single JAX client to orchestrate workloads across multiple large TPU slices, spanning thousands of TPU chips.
* [TPUs](https://cloud.google.com/tpu/docs/v6e) are Google's custom-developed accelerator for ML and AI models built using frameworks such as TensorFlow, PyTorch, and JAX. TPU v6e is Cloud TPU's latest generation AI accelerator.


## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Pathways Job API - Helm chart is used to configure and deploy the
  [Pathways Job](https://github.com/google/pathways-job).
  This job encapsulates the inference of the Llama-4-Maverick-17B-128E model using JetStream MaxText Engine.
  The chart generates the job's manifest, which adhere to best practices for using TPU v6e
  with Google Kubernetes Engine (GKE).

## Prerequisites

1. Verify that your project has enough quota in your region of choice for:  
   * A Cloud TPU slice, for example v6e-32 (`TPUS_PER_TPU_FAMILY`)  
   * A GCE VM with the high memory (eg: `n2d-highmem-64`)
2. Required IAM Permissions  
   Make sure that you have the following roles on the project:   
   * Compute Admin (`roles/compute.admin`)  
   * Kubernetes Engine Admin (`roles/container.admin`)  
   * Storage Admin (`roles/storage.admin`)  
   * Logging Admin (`roles/logging.admin`)  
   * Monitoring Admin (`roles/monitoring.admin`)  
   * Artifact Registry Writer (`roles/artifactregistry.writer`)  
   * Service Account Admin (`roles/iam.serviceAccountAdmin`)  
   * Project IAM Admin (`roles/resourcemanager.projectIamAdmin`)  
3. Access to Pathways Container Images  
   You run a Pathways cluster on GKE in one of the Pathways container images. To access the Pathways container images, the service account used by the cluster must be allowlisted. Reach out to your GCP account team to request access.  
4. Access to Llama models on Hugging Face.  
   To access the [Llama-4-Maverick-17B-128E model](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E) through Hugging Face, you'll need a Hugging Face token. Ensure that you also sign the community license agreement and get gated access to the Meta models. Follow these steps to generate a new token if you don't have one already:
   - Create a [Hugging Face account](https://huggingface.co/), if you don't already have one.
   - Click Your **Profile > Settings > Access Tokens**.
   - Select **New Token**.
   - Specify a Name and a Role of at least `Read`.
   - Select **Generate a token**.
   - Copy the generated token to your clipboard.


## Setup your local environment

We recommend running this recipe from [Cloud Shell](https://console.cloud.google.com/?cloudshell=true) or a client workstation with the following pre-installed:

* [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)  
* [Helm](https://helm.sh/docs/intro/install/)  
* [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)

Install [xpk](https://github.com/AI-Hypercomputer/xpk) toolkit that lets you create pre-configured GKE clusters that support Pathways-based workloads

```bash
git clone https://github.com/AI-Hypercomputer/xpk.git ~/xpk
cd ~/xpk
make install && export PATH=$PATH:$PWD/bin
```

### Clone the recipe

From your client, clone the [`tpu-recipes`](https://github.com/AI-Hypercomputer/tpu-recipes) repository and set a reference to the recipe folder.

```bash
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/inference/trillium/JetStream-Maxtext/Llama-4-Maverick-17B-128E
```

### Configure environment settings

Define the following environment variables with values appropriate to your workload: 

```bash
# Required variables to be set
export PROJECT_ID=<PROJECT_ID>
export REGION=<REGION>
export CLUSTER_NAME=<CLUSTER_NAME>
export CLUSTER_ZONE=<CLUSTER_ZONE>
export GCS_BUCKET=<GCS_BUCKET>
export TPU_RESERVATION=<TPU_RESERVATION>

# Required variables with default values
export TPU_TYPE=v6e-32
export NUM_SLICES=1
export ARTIFACT_REGISTRY_REPO_NAME=jetstream-maxtext-ar
export ARTIFACT_REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_NAME}
export JETSTREAM_MAXTEXT_IMAGE=jetstream-maxtext
export JETSTREAM_MAXTEXT_VERSION=latest
```

Following are required variables that must be set:

- `<PROJECT_ID>`: your Google Cloud project ID  
- `<REGION>`: the region where you want to run Cloud Build  
- `<CLUSTER_NAME>`: the name of your GKE cluster  
- `<CLUSTER_ZONE>`: the zone where your cluster is located  
- `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the gs:// prefix  
- `<TPU_RESERVATION>`: the name of the TPU reservation

Following are required variables with default values already set:

- `TPU_TYPE`: TPU accelerator type supported by TPU v6e. Refer to the [supported list](https://cloud.google.com/tpu/docs/v6e#configurations).   
- `NUM_SLICES`: The number of slices to use  
- `ARTIFACT_REGISTRY`: the full name of your Artifact Registry in the following format: *LOCATION*\-docker.pkg.dev/*PROJECT\_ID*/*REPOSITORY*  
- `JETSTREAM_MAXTEXT_IMAGE`: the name of the JetStream MaxText image  
- `JETSTREAM_MAXTEXT_VERSION`: the version of the JetStream MaxText image


Set the default project:

```bash
gcloud config set project $PROJECT_ID
```

## Create GKE Cluster with TPU v6e nodepool using xpk

Use a custom network for better performance as well as to avoid having the default network becoming overloaded. Please refer to the [network performance optimizations](https://cloud.google.com/tpu/docs/v6e-intro/#network_performance_optimizations) for more details.

```bash
export NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1
export NETWORK_FW_NAME_1=${NETWORK_NAME_1}-fw-1

# Use a custom network for better performance as well as avoid the default network to be overloaded.
gcloud compute networks create ${NETWORK_NAME_1} --mtu=8896 --project=${PROJECT_ID} --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create ${NETWORK_FW_NAME_1} --network ${NETWORK_NAME_1} --allow tcp,icmp,udp --project=${PROJECT_ID}

# Secondary subnet for multinic experience. Need custom ip routing to be different from first network’s subnet.
export NETWORK_NAME_2=${CLUSTER_NAME}-privatenetwork-4
export SUBNET_NAME_2=${CLUSTER_NAME}-privatesubnet-4
export FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-4
export ROUTER_NAME=${CLUSTER_NAME}-network-4
export NAT_CONFIG=${CLUSTER_NAME}-natconfig-4

# Create networks
gcloud compute networks create "${NETWORK_NAME_2}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project=${PROJECT_ID}

# Create subnets
gcloud compute networks subnets create "${SUBNET_NAME_2}" --network="${NETWORK_NAME_2}" --range=10.10.0.0/18 --region="${REGION}" --project=${PROJECT_ID}

# Create Firewall rules
gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME_2}" --allow tcp,icmp,udp --project="${PROJECT_ID}"

# Create router
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT_ID}" \
  --network="${NETWORK_NAME_2}" \
  --region="${REGION}"

# Create NAT
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT_ID}" \
  --enable-logging
```

Create GKE cluster using xpk toolkit with custom network and TPU v6e nodepool

```bash
export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1} --subnetwork=${NETWORK_NAME_1} --scopes cloud-platform"

export NODE_POOL_ARGUMENTS="--additional-node-network network=${NETWORK_NAME_2},subnetwork=${SUBNET_NAME_2} --scopes cloud-platform --workload-metadata=GCE_METADATA --placement-type=COMPACT"

python3 ~/xpk/xpk.py cluster create \
  --cluster $CLUSTER_NAME \
  --default-pool-cpu-machine-type=$CLUSTER_CPU_MACHINE_TYPE \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --zone=${CLUSTER_ZONE} \
  --project=${PROJECT_ID} \
  --reservation=${TPU_RESERVATION} \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
  --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}"
```

## Create a Cloud Storage bucket to store checkpoints and temporary files

Create a Cloud Storage bucket to store model checkpoint, Pathways temporary files like compilation cache. It's recommended to create a bucket in the same region as the TPU nodepool is located.

```bash
gcloud storage buckets create gs://$GCS_BUCKET --location=$REGION
```

## Build JetStream/MaxText container image to deploy the workload

### Create Artifact Registry repository to store Docker images

```bash
gcloud artifacts repositories create ${ARTIFACT_REGISTRY_REPO_NAME} \
      --repository-format=docker \
      --location=${REGION} \
      --description="Repository for JetStream/MaxText container images" \
      --project=${PROJECT_ID}
```

### Configure Docker to authenticate to Artifact Registry

[Configure Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper) to authenticate to Artifact Registry to pull the allowlisted Pathways images

```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### Build and push the Docker container image to Artifact Registry

To build the container, submit a Cloud Build job to build and push the container image running the following command from your client:

```bash
cd $RECIPE_ROOT/docker
gcloud builds submit \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --config cloudbuild.yml \
  --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY,_JETSTREAM_MAXTEXT_IMAGE=$JETSTREAM_MAXTEXT_IMAGE,_JETSTREAM_MAXTEXT_VERSION=$JETSTREAM_MAXTEXT_VERSION \
  --timeout "2h" \
  --machine-type=e2-highcpu-32 \
  --disk-size=1000 \
  --quiet \
  --async
```

This command outputs the `BUILD ID`. You can monitor the build progress by streaming the logs for the `BUILD ID`. To do this, run the following command with `<BUILD_ID>` replaced with your build ID.

```bash
BUILD_ID=<BUILD_ID>
gcloud beta builds log $BUILD_ID --region=$REGION
```

### Get cluster credentials

From your client, get the credentials for your cluster.

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

## Multihost Inference of Llama-4-Maverick-17B-128E

The recipe serves Llama-4-Maverick-17B-128E model using JetStream MaxText Engine on `v6e-32` mulithost slice of TPU v6e Trillium

To start the inference, the recipe launches JetStream MaxText Engine that does the following steps:
1. Downloads the full Llama-4-Maverick-17B-128E model PyTorch checkpoints from [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Original).
2. Convert the model checkpoints from PyTorch format to JAX Orbax format.
3. Start the JetStream MaxText Engine server.
3. Inference is ready to respond to requests and run benchmarks

The recipe uses the helm chart to run the above steps.

1. Create Kubernetes Secret with a Hugging Face token to enable the job to download the model checkpoints.

    ```bash
    export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
    ```

    ```bash
    kubectl create secret generic hf-secret \
    --from-literal=hf_api_token=${HF_TOKEN} \
    --dry-run=client -o yaml | kubectl apply -f -
    ```

2. Convert the checkpoint from PyTorch to Orbax 
    This job converts the checkpoint from PyTorch format to JAX Orbax format and unscans it for performant serving. This unscanned checkpoint is then stored in the mounted GCS bucket so that it can be used by the TPU nodepool to bring up the JetStream serve in the next step.

    ```bash
    cd $RECIPE_ROOT
    helm install -f values.yaml \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set clusterName=$CLUSTER_NAME \
    --set pathwaysDir="gs://${GCS_BUCKET}" \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${JETSTREAM_MAXTEXT_IMAGE} \
    --set job.image.tag=${JETSTREAM_MAXTEXT_VERSION} \
    --set convert_hf_ckpt=false \
    $USER-serving-llama4-model \
    $RECIPE_ROOT/prepare-model
    ```
    
    Run the following to verify if the job has been completed.
      ```bash
       kubectl get job/$USER-serving-llama4-model-convert-ckpt

       NAME                                     STATUS    COMPLETIONS   DURATION   AGE
       user-serving-llama4-model-convert-ckpt   Running   1/1           26m        26m 
      ```

    Uninstall the helm chart once done
    ```bash
    helm uninstall $USER-serving-llama4-model
    ```

3. Install the `PathwaysJob API` on your GKE cluster
    ```bash
    kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.0/manifests.yaml
    kubectl apply --server-side -f https://github.com/google/pathways-job/releases/download/v0.1.0/install.yaml
    ```

4. Bring up the Pathways JetStream MaxText server
    ```bash
    cd $RECIPE_ROOT
    /usr/local/bin/helm/helm template -f values.yaml \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set pathwaysDir="gs://${GCS_BUCKET}"
    --set clusterName=$CLUSTER_NAME \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${JETSTREAM_MAXTEXT_IMAGE} \
    --set job.image.tag=${JETSTREAM_MAXTEXT_VERSION} \
    $USER-serving-llama4-model \
    $RECIPE_ROOT/serve-model --dry-run --debug
    ```

    Verify if the deployment has started by running

    ```bash
      kubectl get pods

      NAME                               READY   STATUS    RESTARTS   AGE
      pathways-pathways-head-0-0-f7zct   4/4     Running   0          118s
      pathways-worker-0-0-pfld6          1/1     Running   0          108s
      pathways-worker-0-1-gdxvm          1/1     Running   0          108s
      pathways-worker-0-10-qnttk         1/1     Running   0          106s
      pathways-worker-0-11-br6jn         1/1     Running   0          106s
      pathways-worker-0-12-zqtkk         1/1     Running   0          106s
      pathways-worker-0-13-5bl7p         1/1     Running   0          106s
      pathways-worker-0-14-72nmt         1/1     Running   0          106s
      pathways-worker-0-15-vxlfg         1/1     Running   0          106s
      pathways-worker-0-2-wb9sn          1/1     Running   0          107s
      pathways-worker-0-3-ccgd5          1/1     Running   0          107s
      pathways-worker-0-4-wwmvd          1/1     Running   0          107s
      pathways-worker-0-5-rds6d          1/1     Running   0          107s
      pathways-worker-0-6-m2mzz          1/1     Running   0          107s
      pathways-worker-0-7-kq2pz          1/1     Running   0          107s
      pathways-worker-0-8-j64dp          1/1     Running   0          107s
      pathways-worker-0-9-6brpd          1/1     Running   0          106s
    ```

    The server bring up takes ~20 mins with GCS. You can verify if it is ready by running:
      ```bash
      HEAD_POD=$(kubectl get pods | grep pathways--pathways-head | awk '{print $1}')
      kubectl logs -f ${HEAD_POD} -c jetstream


    
      ```

4. Stop the server and clean up the resources after completion by following the steps in the [Cleanup](#cleanup) section.
    

### Cleanup

To clean up the resources created by this recipe, complete the following steps:

1. Uninstall the helm chart.

    ```bash
    helm uninstall $USER-serving-llama4-model
    ```

2. Delete the Kubernetes Secret.

    ```bash
    kubectl delete secret hf-secret
    ```
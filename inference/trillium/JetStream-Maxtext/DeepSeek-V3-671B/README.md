# DeepSeek R1/V3 Multi-host Inference on TPU v6e with JetStream, MaxText and Pathways on Cloud with GKE Cluster

This recipe outlines the steps to benchmark [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) or [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) 671B model using [JetStream](https://github.com/AI-Hypercomputer/JetStream/tree/main) \+ [MaxText](https://github.com/AI-Hypercomputer/maxtext) inference engine deployed on a GKE cluster with multi-host [TPU v6e slices](https://cloud.google.com/kubernetes-engine) utilizing [Pathways on Cloud](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro).

* [Jetstream](https://github.com/AI-Hypercomputer/JetStream) is a throughput and memory-optimized engine for LLM inference on XLA devices, primarily TPUs written in JAX.  
* [MaxText](https://github.com/AI-Hypercomputer/maxtext) is an open-source LLM project by Google, written in JAX and designed to be highly performant and scalable, running efficiently on Google Cloud TPUs and GPUs.   
* [Pathways](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro) is a system that simplifies large-scale ML computations by enabling a single JAX client to orchestrate workloads across multiple large TPU slices, spanning thousands of TPU chips.  
* [TPUs](https://cloud.google.com/tpu/docs/v6e) are Google's custom-developed accelerator for ML and AI models built using frameworks such as TensorFlow, PyTorch, and JAX. TPU v6e is Cloud TPU's latest generation AI accelerator.

## Outline

1. [Ensure prerequisites are met.](#prerequisites)
2. [Setup development environment.](#setup-your-local-environment)
3. [Provision a GKE Cluster with TPU v6e and CPU nodepools](#create-gke-cluster-with-tpu-v6e-nodepool-using-xpk)
4. [Configure service account for access](#configure-a-service-account-for-access)  
5. [Create container image with dependencies](#build-jetstreammaxtext-container-image-to-deploy-the-workload)  
6. [Checkpoint conversion](#checkpoint-conversion)  
   - Download model weights from HuggingFace
   - Convert Hugging Face checkpoint from FP8 to BF16
   - Convert Hugging Face BF16 checkpoint to MaxText compatible checkpoint
7. [Deploy JetStream and Pathways](#deploy-jetstream-and-pathways)  
8. [Run MMLU benchmark](#run-mmlu-benchmark)

## Prerequisites

1. Verify that your project has enough quota in your region of choice for:  
   * A Cloud TPU slice, for example v6e-64 (`TPUS_PER_TPU_FAMILY`)  
   * Compute Engine API quota for M1 machine configuration for 160 chips (`M1_CPUS`)
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
3. Access to Pathways Container Images.
   * You run a Pathways cluster on GKE in one of the [Pathways container images](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro#pathways-components).
4. Access to DeepSeek models on Hugging Face.  
   To access the [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) or [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) model through Hugging Face, you'll need a Hugging Face token. Follow these steps to generate a new token if you don't have one already:  
   * Create a [Hugging Face account](https://huggingface.co/), if you don't already have one.  
   * Click Your **Profile \> Settings \> Access Tokens**.  
   * Select **New Token**.  
   * Specify a Name and a Role of at least Read.  
   * Select **Generate a token**.  
   * Copy the generated token to your clipboard.

## Setup your local environment

We recommend running this recipe from [Cloud Shell](https://console.cloud.google.com/?cloudshell=true) or a client workstation with the following pre-installed:

* [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)  
* [Helm](https://helm.sh/docs/intro/install/)  
* [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)

Install [xpk](https://github.com/AI-Hypercomputer/xpk) toolkit that lets you create pre-configured GKE clusters that support Pathways-based workloads

``` bash
git clone https://github.com/AI-Hypercomputer/xpk.git ~/xpk
cd ~/xpk
make install && export PATH=$PATH:$PWD/bin
```

### Clone the recipe

From your client, clone the [`tpu-recipes`](https://github.com/AI-Hypercomputer/tpu-recipes) repository and set a reference to the recipe folder.

``` bash
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/inference/trillium/JetStream-Maxtext/DeepSeek-V3-671B
```

### Configure environment settings

Define the following environment variables with values appropriate to your workload: 

``` bash
# Required variables to be set
export PROJECT_ID=<PROJECT_ID>
export REGION=<REGION>
export CLUSTER_NAME=<CLUSTER_NAME>
export CLUSTER_ZONE=<CLUSTER_ZONE>
export GCS_BUCKET=<GCS_BUCKET>
export TPU_RESERVATION=<TPU_RESERVATION>

# Required variables with default values
export TPU_TYPE=v6e-64
export NUM_SLICES=1
export CLUSTER_CPU_MACHINE_TYPE=n2d-standard-32
export CLUSTER_CKPT_NODEPOOL_NAME=ckpt-conversion-node-pool-0
export CLUSTER_CKPT_NODE_MACHINE_TYPE=m1-ultramem-160
export CLUSTER_CKPT_NODE_REGION=us-east4
export CLUSTER_CKPT_NODE_DISK_SIZE=3000
export CLUSTER_CKPT_NUM_NODES=1
export ARTIFACT_REGISTRY_REPO_NAME=jetstream-maxtext-ar
export ARTIFACT_REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_NAME}
export JETSTREAM_MAXTEXT_IMAGE=jetstream-maxtext
export JETSTREAM_MAXTEXT_VERSION=latest
export HF_MODEL_NAME="deepseek-ai/DeepSeek-R1"
export MODEL_NAME=deepseek3-671b
export GCS_CKPT_PATH_BF16=gs://${GCS_BUCKET}/models/${MODEL_NAME}/bf16
export GCS_CKPT_PATH_UNSCANNED=gs://${GCS_BUCKET}/models/${MODEL_NAME}/unscanned
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
- `CLUSTER_CPU_MACHINE_TYPE`: The CPU nodepool machine type  
- `CLUSTER_CKPT_NODEPOOL_NAME`: The name of CPU nodepool used for checkpoint conversion  
- `CLUSTER_CKPT_NODE_MACHINE_TYPE`: The machine type of CPU nodepool used for checkpoint conversion  
- `CLUSTER_CKPT_NODE_DISK_SIZE`: The disk size of CPU nodepool used for checkpoint conversion. For this recipe, a minimum of 3TB disk size is suggested.  
- CLUSTER\_CKPT\_NUM\_NODES  
- `ARTIFACT_REGISTRY`: the full name of your Artifact Registry in the following format: *LOCATION*\-docker.pkg.dev/*PROJECT\_ID*/*REPOSITORY*  
- `JETSTREAM_MAXTEXT_IMAGE`: the name of the JetStream MaxText image  
- `JETSTREAM_MAXTEXT_VERSION`: the version of the JetStream MaxText image

Set the default project:

``` bash
gcloud config set project $PROJECT_ID
```

## Create GKE Cluster with TPU v6e nodepool using xpk

Use a custom network for better performance as well as to avoid having the default network becoming overloaded. Please refer to the [network performance optimizations](https://cloud.google.com/tpu/docs/v6e-intro/#network_performance_optimizations) for more details.

``` bash
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

``` bash
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

``` bash
gcloud storage buckets create gs://$GCS_BUCKET --location=$REGION
```

## Configure a service account for access

Configure a Kubernetes service account to act as an IAM service account. 

* Create an IAM service account for your application:

``` bash
gcloud iam service-accounts create jetstream-pathways
```

* Add an IAM policy binding for your IAM service account to manage Cloud Storage. This is to access the storage bucket where your checkpoint will be stored:

``` bash
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member "serviceAccount:jetstream-pathways@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role roles/storage.objectUser

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member "serviceAccount:jetstream-pathways@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role roles/storage.insightsCollectorService
```

* Annotate the Kubernetes service account with the email address of the IAM service account. 

``` bash
kubectl annotate serviceaccount default \
iam.gke.io/gcp-service-account=jetstream-pathways@${PROJECT_ID}.iam.gserviceaccount.com
```

## Build JetStream/MaxText container image to deploy the workload

### Create Artifact Registry repository to store Docker images

``` bash
gcloud artifacts repositories create ${ARTIFACT_REGISTRY_REPO_NAME} \
      --repository-format=docker \
      --location=${REGION} \
      --description="Repository for JetStream/MaxText container images" \
      --project=${PROJECT_ID}
```

### Configure Docker to authenticate to Artifact Registry

[Configure Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper) to authenticate to Artifact Registry to pull the allowlisted Pathways images

``` bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### Build and push the Docker container image to Artifact Registry

To build the container, submit a Cloud Build job to build and push the container image running the following command from your client:

``` bash
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

``` bash
BUILD_ID=<BUILD_ID>
gcloud beta builds log $BUILD_ID --region=$REGION
```

## Checkpoint conversion

This step requires an `m1-ultramem-160` (memory-optimized) machine with 3TB of storage that can be run. The recipe uses a [Cloud Batch job](https://cloud.google.com/batch/docs/get-started) to run the conversion.

The following job performs following steps:
- Downloads DeepSeek V3 or DeepSeek R1 (defined by `HF_MODEL_NAME`) weights from HuggingFace.
- Convert Hugging Face checkpoint weights from FP8 to BF16.
- Convert BF16 weights  to MaxText compatible format (unscanned checkpoint) for efficient serving. 

Submit Cloud Batch job. This step can take >2 hours.

``` bash
cd $RECIPE_ROOT/prepare-model
gcloud batch jobs submit convert-ckpt-to-unscanned-$(date +%Y%m%d-%H%M%S) \
  --project ${PROJECT_ID} \
  --location ${CLUSTER_CKPT_NODE_REGION} \
  --config - <<EOF
$(envsubst < batch_job.yaml)
EOF
```

You can monitor the progress of the job on Cloud Console from the [job list page](https://console.cloud.google.com/batch/jobs). Click the name of a job to view the job details including logs.

## Deploy JetStream and Pathways

Get cluster credentials and connect to the GKE cluster 

``` bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
```

### Deploy LeaderWorkerSet (LWS) API

LWS is a custom resource designed for deploying and managing stateful, distributed applications, particularly those with a leader-worker architecture. It's especially well-suited for AI/ML workloads where a large model is sharded and served across multiple devices on multiple nodes.

``` bash
VERSION=v0.6.0
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/lws/releases/download/${VERSION}/manifests.yaml"
```

Validate that the LeaderWorkerSet controller is running in the lws-system namespace:

``` bash
kubectl get pod -n lws-system
```

The output should be similar to the following:

``` bash
NAME                          READY   STATUS    RESTARTS    AGE
lws-controller-manager-abcd   2/2     Running   0           12d
lws-controller-manager-efgh   2/2     Running   0           12d
```

### Deploy the workload manifest

This step starts the JetStream inference engine and Pathways.  

![](./pathways.png)

Note: Each Host corresponds to a TPU VM

Install the helm chart to serve the unscanned model checkpoint and bring up the JetStream MaxText Engine server on Pathways system

``` bash
cd $RECIPE_ROOT
helm install -f values.yaml \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set clusterName=$CLUSTER_NAME \
--set job.jax_tpu_image.repository=${ARTIFACT_REGISTRY}/${JETSTREAM_MAXTEXT_IMAGE} \
--set job.jax_tpu_image.tag=${JETSTREAM_MAXTEXT_VERSION} \
--set maxtext_config.load_parameters_path=${GCS_CKPT_PATH_UNSCANNED}/0/items \
jetstream-pathways \
$RECIPE_ROOT/serve-model
```

You can check the status of the job via:

``` bash
kubectl get pods
```

``` bash
NAME                      READY   STATUS    RESTARTS   AGE
jetstream-pathways-0      4/4     Running   0          11h
jetstream-pathways-0-1    1/1     Running   0          11h
jetstream-pathways-0-10   1/1     Running   0          11h
jetstream-pathways-0-11   1/1     Running   0          11h
jetstream-pathways-0-12   1/1     Running   0          11h
jetstream-pathways-0-13   1/1     Running   0          11h
jetstream-pathways-0-14   1/1     Running   0          11h
jetstream-pathways-0-15   1/1     Running   0          11h
jetstream-pathways-0-16   1/1     Running   0          11h
jetstream-pathways-0-2    1/1     Running   0          11h
jetstream-pathways-0-3    1/1     Running   0          11h
jetstream-pathways-0-4    1/1     Running   0          11h
jetstream-pathways-0-5    1/1     Running   0          11h
jetstream-pathways-0-6    1/1     Running   0          11h
jetstream-pathways-0-7    1/1     Running   0          11h
jetstream-pathways-0-8    1/1     Running   0          11h
jetstream-pathways-0-9    1/1     Running   0          11h
```

To view the logs for the deployment, run

``` bash
kubectl logs -f jetstream-pathways-0 -c jax-tpu
```

You should see the following in the logs When the server is up and running:

``` bash
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9999 (Press CTRL+C to quit)
```

After the server is up and running, you can SSH into one of the workers and prompt the model:

``` bash
kubectl exec -it jetstream-pathways-0 -c jax-tpu -- /bin/bash
```

``` bash
curl --request POST --header "Content-type: application/json" -s localhost:8000/generate --data '{
    "prompt": "What are the top 5 programming languages",                                                       
    "max_tokens": 200
}'
```

## Run MMLU benchmark

After the JetStream server is up and running, SSH into one of the \`jax-tpu\` workers

``` bash
kubectl exec -it jetstream-pathways-0 -c jax-tpu -- /bin/bash
```

Download full MMLU set

``` bash
LOCAL_DIR=/data
mkdir -p ${LOCAL_DIR}/mmlu
cd ${LOCAL_DIR}/mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P ${LOCAL_DIR}/mmlu
tar -xvf data.tar
```

Run the benchmarking script

``` bash
python3 /JetStream/benchmarks/benchmark_serving.py \
  --use-hf-tokenizer=True \
  --use-chat-template=False \
  --hf-access-token=$HF_TOKEN \
  --tokenizer=deepseek-ai/DeepSeek-R1 \
  --num-prompts 14037 \
  --dataset mmlu \
  --dataset-path ${LOCAL_DIR}/mmlu/data/test \
  --request-rate 0 \
  --warmup-mode sampled \
  --save-request-outputs \
  --num-shots=5 \
  --run-eval True \
  --model=deepseek3-671b \
  --save-result
```

The results will be printed at the end and the accuracy and throughput numbers will be saved by default to `JetStream-0.0qps-deepseek3-671b-*.json` and inputs, model predictions, and outputs will be saved by default to `$HOME/tmp/request-outputs.json`

You can also run a lighter weight experiment to quickly confirm the numbers are good by changing:

``` bash
python3 /JetStream/benchmarks/benchmark_serving.py \
  --use-hf-tokenizer=True \
  --use-chat-template=False \
  --tokenizer=deepseek-ai/DeepSeek-R1 \
  --dataset mmlu \
  --dataset-path /JetStream/benchmarks/mmlu_test_dataset \
  --num-prompts 200 \
  --request-rate 0 \
  --warmup-mode sampled \
  --save-request-outputs \
  --num-shots=5 \
  --run-eval True \
  --model=deepseek3-671b \
  --save-result
```

## Cleanup

* Delete GKE cluster with v6e TPU nodepool

``` bash
gcloud container clusters delete $CLUSTER_NAME --zone $CLUSTER_ZONE
```

* Delete Cloud Storage buckets

``` bash
gcloud storage buckets delete ${GCS_BUCKET}
```

* Delete VPC networks, subnets, routers, gateways and firewall policies used by the GKE cluster

``` bash
# --- Delete Resources for Network 2 (${NETWORK_NAME_2}) ---
# Order: NAT -> Router -> Firewall Rule -> Subnet -> Network

echo "Deleting NAT Gateway: ${NAT_CONFIG}..."
gcloud compute routers nats delete ${NAT_CONFIG} \
  --router=${ROUTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --quiet || echo "Warning: Failed to delete NAT ${NAT_CONFIG} (may already be deleted or dependencies exist)."

echo "Deleting Router: ${ROUTER_NAME}..."
gcloud compute routers delete ${ROUTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --quiet || echo "Warning: Failed to delete Router ${ROUTER_NAME} (may already be deleted or dependencies exist)."

echo "Deleting Firewall Rule: ${FIREWALL_RULE_NAME}..."
gcloud compute firewall-rules delete ${FIREWALL_RULE_NAME} \
  --project=${PROJECT_ID} \
  --quiet || echo "Warning: Failed to delete Firewall Rule ${FIREWALL_RULE_NAME} (may already be deleted)."

echo "Deleting Subnet: ${SUBNET_NAME_2}..."
gcloud compute networks subnets delete ${SUBNET_NAME_2} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --quiet || echo "Warning: Failed to delete Subnet ${SUBNET_NAME_2} (may already be deleted or dependencies exist)."

echo "Deleting Network: ${NETWORK_NAME_2}..."
gcloud compute networks delete ${NETWORK_NAME_2} \
  --project=${PROJECT_ID} \
  --quiet || echo "Warning: Failed to delete Network ${NETWORK_NAME_2} (may already be deleted or dependencies exist)."

# --- Delete Resources for Network 1 (${NETWORK_NAME_1}) ---
# Order: Firewall Rule -> Network (Auto-created subnets are deleted with the network if empty)

echo "Deleting Firewall Rule: ${NETWORK_FW_NAME_1}..."
gcloud compute firewall-rules delete ${NETWORK_FW_NAME_1} \
  --project=${PROJECT_ID} \
  --quiet || echo "Warning: Failed to delete Firewall Rule ${NETWORK_FW_NAME_1} (may already be deleted)."

echo "Deleting Network: ${NETWORK_NAME_1}..."
gcloud compute networks delete ${NETWORK_NAME_1} \
  --project=${PROJECT_ID} \
  --quiet || echo "Warning: Failed to delete Network ${NETWORK_NAME_1} (may already be deleted or dependencies exist)."
```
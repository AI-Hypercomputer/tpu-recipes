# Inference benchmark of Llama-4-Scout-17B-16E with JetStream MaxText Engine on TPU v6e (Trillium)

This recipe outlines the steps to benchmark the inference of a Llama-4-Scout-17B-16E model using [JetStream](https://github.com/AI-Hypercomputer/JetStream/tree/main) with [MaxText](https://github.com/AI-Hypercomputer/maxtext) engine on an [TPU v6e Node pool](https://cloud.google.com/kubernetes-engine) with a single node.

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Kubernetes Deployment - Helm chart is used to configure and deploy the
  [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment).
  This deployment encapsulates the inference of the Llama-4-Scout-17B-16E model using JetStream MaxText Engine.
  The chart generates the job's manifest, which adhere to best practices for using TPU v6e
  with Google Kubernetes Engine (GKE).

## Prerequisites

To prepare the required environment, complete the following steps:

1. Create a GKE Cluster with TPU v6e Node pool. You can choose to use either GKE Autopilot or GKE Standard, but for this recipe, we will use GKE Standard.
  ```bash
  gcloud container clusters create <CLUSTER_NAME> \
      --project=<PROJECT_ID> \
      --zone=<ZONE> \
      --cluster-version=1.31.5-gke.1023000 \
      --workload-pool=<PROJECT_ID>.svc.id.goog \
      --addons GcsFuseCsiDriver
  ```

2. Create a TPU v6e slice node pool in the cluster.
  ```
  gcloud container node-pools create tpunodepool \
      --zone=<ZONE> \
      --num-nodes=1 \
      --machine-type=ct6e-standard-8t \
      --cluster=<CLUSTER_NAME> \
      --enable-autoscaling --total-min-nodes=1 --total-max-nodes=2
  ```

3. Create a CPU nodepool with a high memory machine to convert the checkpoint
  ```
  gcloud container node-pools create cpunodepool \
      --zone=<ZONE> \
      --num-nodes=1 \
      --machine-type=n2-highmem-80 \
      --cluster=<CLUSTER_NAME> \
      --image_type=COS_CONTAINERD \
      --disk-type=pd-balanced
      --disk-size=100
  ```

GKE creates the following resources for the recipe:

- A GKE Standard cluster that uses Workload Identity Federation for GKE and has Cloud Storage FUSE CSI driver enabled.
- A TPU Trillium node pool with a `ct6e-standard-8t` machine type. This node pool has one node, eight TPU chips, and autoscaling enabled
- A CPU high memory nodepool with a `n2-highmem-80` machine type. This node pool has one node with 640G of memory for converting the checkpoint

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - A TPU Trillium node pool with a `ct6e-standard-8t` machine type. 
    - Topology-aware scheduling enabled
- An Artifact Registry repository to store the Docker image.
- A Google Cloud Storage (GCS) bucket to store results.
  *Important: This bucket must be in the same region as the GKE cluster*.
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl
- To access the [Llama-4-Scout-17B-16E model](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) through Hugging Face, you'll need a Hugging Face token. Follow these steps to generate a new token if you don't have one already:
   - Create a [Hugging Face account](https://huggingface.co/), if you don't already have one.
   - Click Your **Profile > Settings > Access Tokens**.
   - Select **New Token**.
   - Specify a Name and a Role of at least `Read`.
   - Select **Generate a token**.
   - Copy the generated token to your clipboard.


## Run the recipe

### Launch Cloud Shell

In the Google Cloud console, start a [Cloud Shell Instance](https://console.cloud.google.com/?cloudshell=true).

### Configure environment settings

From your client, complete the following steps:

1. Set the environment variables to match your environment:

  ```bash
  export PROJECT_ID=<PROJECT_ID>
  export REGION=<REGION>
  export CLUSTER_REGION=<CLUSTER_REGION>
  export CLUSTER_NAME=<CLUSTER_NAME>
  export GCS_BUCKET=<GCS_BUCKET>
  export ARTIFACT_REGISTRY=<ARTIFACT_REGISTRY>
  export JETSTREAM_MAXTEXT_IMAGE=jetstream-maxtext
  export JETSTREAM_MAXTEXT_VERSION=latest
  ```
  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<REGION>`: the region where you want to run Cloud Build
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
  - `<ARTIFACT_REGISTRY>`: the full name of your Artifact
    Registry in the following format: *LOCATION*-docker.pkg.dev/*PROJECT_ID*/*REPOSITORY*
  - `<JETSTREAM_MAXTEXT_IMAGE>`: the name of the JetStream MaxText image
  - `<JETSTREAM_MAXTEXT_VERSION>`: the version of the JetStream MaxText image

1. Set the default project:

  ```bash
  gcloud config set project $PROJECT_ID
  ```

### Get the recipe

From your client, clone the `tpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/inference/trillium/JetStream-Maxtext/Llama-4-Scout-17B-16E
```

### Get cluster credentials

From your client, get the credentials for your cluster.

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Build and push a docker container image to Artifact Registry

To build the container, complete the following steps from your client:

1. Use Cloud Build to build and push the container image.

    ```bash
    cd $RECIPE_ROOT/docker
    gcloud builds submit --region=global \
        --config cloudbuild.yml \
        --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY,_JETSTREAM_MAXTEXT_IMAGE=$JETSTREAM_MAXTEXT_IMAGE,_JETSTREAM_MAXTEXT_VERSION=$JETSTREAM_MAXTEXT_VERSION \
        --timeout "2h" \
        --machine-type=e2-highcpu-32 \
        --disk-size=1000 \
        --quiet \
        --async
    ```
  This command outputs the `build ID`.

2. You can monitor the build progress by streaming the logs for the `build ID`.
   To do this, run the following command.

   Replace `<BUILD_ID>` with your build ID.

   ```bash
   BUILD_ID=<BUILD_ID>

   gcloud beta builds log $BUILD_ID --region=$REGION
   ```

## Single TPU v6e Node Inference of Llama-4-Scout-17B-16E

The recipe serves Llama-4-Scout-17B-16E model using JetStream MaxText Engine on a single TPU v6e node.

To start the inference, the recipe launches JetStream MaxText Engine that does the following steps:
1. Downloads the full Llama-4-Scout-17B-16 model PyTorch checkpoints from [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Original).
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
    --set job.image.repository=${ARTIFACT_REGISTRY}/${JETSTREAM_MAXTEXT_IMAGE} \
    --set job.image.tag=${JETSTREAM_MAXTEXT_VERSION} \
    --set convert_hf_ckpt=true \
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

3. Bring up the JetStream MaxText Engine server
    ```bash
    cd $RECIPE_ROOT
    helm install -f values.yaml \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set clusterName=$CLUSTER_NAME \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${JETSTREAM_MAXTEXT_IMAGE} \
    --set job.image.tag=${JETSTREAM_MAXTEXT_VERSION} \
    $USER-serving-llama4-model \
    $RECIPE_ROOT/serve-model
    ```

    Verify if the deployment has started by running

    ```bash
    kubectl get deployment/$USER-serving-llama4-model-serving
    ```

    Once the deployment has started, you'll see logs similar to:
      ```bash
        Loading decode params from /gcs/meta-llama/Llama-4-Scout-17B-16E-Original/output/unscanned_ckpt/checkpoints/0/items
        restoring params from /gcs/meta-llama/Llama-4-Scout-17B-16E-Original/output/unscanned_ckpt/checkpoints/0/items
        WARNING:absl:The transformations API will eventually be replaced by an upgraded design. The current API will not be removed until this point, but it will no longer be actively worked on.

        Memstats: After load_params:
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_0(process=0,(0,0,0,0))
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_1(process=0,(1,0,0,0))
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_2(process=0,(0,1,0,0))
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_3(process=0,(1,1,0,0))
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_4(process=0,(0,2,0,0))
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_5(process=0,(1,2,0,0))
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_6(process=0,(0,3,0,0))
                Using (GB) 25.1 / 31.25 (80.320000%) on TPU_7(process=0,(1,3,0,0))

        RAMstats: After load_params:
                Using (GB) 184.14 / 1417.35 (12.991851%) -->  Available:1224.63
        2025-04-16 22:19:57,301 - jetstream.core.server_lib - INFO - Loaded all weights.
        GC tweaked (allocs, gen1, gen2):  60000 20 30
        2025-04-16 22:19:58,102 - jetstream.core.server_lib - INFO - Starting server on port 9000 with 256 threads
        2025-04-16 22:19:58,109 - jetstream.core.server_lib - INFO - Not starting JAX profiler server: False
        2025-04-16 22:19:58,109 - jetstream.core.server_lib - INFO - Server up and ready to process requests on port 9000
      ```

4. To run MMLU, run the following command:

  ```bash
    kubectl exec -it deployment/$USER-serving-llama4-model-serving -- /bin/bash -c "JAX_PLATFORMS=tpu python3 /JetStream/benchmarks/benchmark_serving.py \
    --tokenizer meta-llama/Llama-4-Scout-17B-16E \
    --use-hf-tokenizer 1 \
    --hf-access-token $HF_TOKEN \
    --num-prompts 14037 \
    --dataset mmlu \
    --dataset-path /gcs/mmlu/data/test \
    --request-rate 0 \
    --warmup-mode sampled \
    --save-request-outputs \
    --num-shots=5 \
    --run-eval True \
    --model=llama4-17b-16e \
    --save-result \
    --request-outputs-file-path mmlu_outputs.json
  ```
    
    e. Stop the server and clean up the resources after completion by following the steps in the [Cleanup](#cleanup) section.
    

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
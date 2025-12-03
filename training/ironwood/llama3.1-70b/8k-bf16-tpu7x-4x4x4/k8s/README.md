# Pretrain llama3-1-70b workload on Ironwood GKE clusters with Kubernetes JobSet

This recipe outlines the steps for running a llama3-1-70b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine)
by applying a Kubernetes manifest to deploy a JobSet resource.


## Workload Details

This workload is configured with the following details:

-   Sequence Length: 8192
-   Precision: bfloat16
-   Chips: 64 (4x4x4 topology)

## Prerequisites

This recipe assumes the following prerequisites are met:

-   **GKE Cluster:** A GKE cluster with [JobSet](https://jobset.sigs.k8s.io/docs/installation/) installed and running.
-   **Container Image:** A pre-built container image (such as
    `gcr.io/my-project/my-maxtext-runner:latest`) containing the MaxText
    workload, accessible by the GKE cluster.
-   **Tools:** `gcloud`, `kubectl`, `gke-gcloud-auth-plugin`, and `envsubst`
    installed on your workstation. If `envsubst` is missing, install it with
    `sudo apt-get update && sudo apt-get install -y gettext-base`.
-   **Permissions:** You have permissions to run `kubectl apply` on the target
    cluster and the cluster has permissions to pull the container image.

## Orchestration and deployment tools

For this recipe, the following setup is used:

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Pretraining job configuration and deployment** - A Kubernetes manifest
    (`k8s_manifest.yaml`) is used to define and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the MaxText pretraining workload.

## Training dataset

This recipe uses a mock pretraining dataset provided by the MaxText framework.

## Run the recipe

This recipe uses a Kubernetes manifest (`k8s_manifest.yaml`) to deploy the
workload. The following commands will set the required environment variables,
substitute them into `k8s_manifest.yaml`, and apply the resulting
configuration to your cluster.

### 1. Configure Environment Variables

Open a terminal and set the following environment variables to match your setup.
**Note:** `k8s_manifest.yaml` is in the same directory as this README.

```bash
# Set variables for your environment
export PROJECT_ID=""    # Your GCP project name
export CLUSTER_NAME=""  # The name of your GKE cluster
export ZONE=""          # The zone of your GKE cluster
export BASE_OUTPUT_DIR=""    # e.g., "gs://your-bucket-name/my-base-output-dir"
export WORKLOAD_IMAGE=""   # e.g., "gcr.io/my-project/my-maxtext-runner:latest"

# Set workload name (or modify as needed, make sure its unique in the cluster)
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-llama3-1-70b")-$(date +%Y%m%d-%H%M)"
```

### 2. Run llama3-1-70b Pretraining Workload

Once the environment variables are set, run the following commands to fetch
cluster credentials and deploy the JobSet:

```bash
# Fetch cluster credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}

# Apply the manifest
envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE}' < k8s_manifest.yaml | kubectl apply -n default -f -
```

## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and logs:

```bash
# Check JobSet status
kubectl get jobset -n default ${WORKLOAD_NAME}

# Get the name of the first pod in the JobSet
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')

# Follow the logs of that pod
kubectl logs -f -n default ${POD_NAME}
```

You can also monitor your cluster and TPU usage through the Google Cloud
Console:
`https://console.cloud.google.com/kubernetes/workload/overview?project={PROJECT_ID}`

## Delete resources

### Delete a specific workload

To delete the JobSet created by this recipe, run:

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default
```

## Check results

After the job completes, you can check the results by:

-   Accessing output logs from your job using `kubectl logs`.
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

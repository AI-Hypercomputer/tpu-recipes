# Ironwood (Cloud TPU v7x) pretrain muse-glimmer-30b workload on GKE clusters with Cluster Toolkit (gcluster)

This recipe outlines the steps for running a muse-glimmer-30b TorchTitan pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).

<!--
================================================================================
WARNING: WORKLOAD DETAILS MISSING
Some workload details (Sequence Length, Precision, or Chips) are showing as "N/A".
Please ensure these values are correctly configured in the workload or config.yml.
================================================================================
-->

## Workload Details

This workload is configured with the following details:

-   Sequence Length: N/A
-   Precision: N/A
-   Chips: 64 (4x4x4 topology)

## Prerequisites

To run this recipe, you need the following:

-   **GCP Project Setup:** Ensure you have a GCP project with billing enabled
    and are allowlisted for Ironwood access.
-   **User Project Permissions:** The account used requires the following IAM Roles.
-   **Docker:** Docker must be installed.
-   **Cluster Toolkit:** Follow the steps to install it.
-   **HuggingFace Token:** You will need a HuggingFace token. Please update `<YOUR_HF_TOKEN>` in `run_recipe.sh` or `k8s_manifest.yaml` before running.

## Install Cluster Toolkit and dependencies

Follow the official
[Cluster Toolkit Environment Setup Guide](https://docs.cloud.google.com/cluster-toolkit/docs/setup/configure-environment#install)
to configure your environment and install prerequisites.

## Test environment

This recipe is optimized for and tested with tpu7x-4x4x4.

### Environment Variables for Cluster Creation

The environment variables required for cluster creation and workload execution
are defined at the beginning of the `run_recipe.sh` script. **Before running the
`gcluster job submit` command**, please open `run_recipe.sh` and modify the
`export` statements to set these variables to match your environment. 

-   `PROJECT_ID`: Your GCP project name.
-   `CLUSTER_NAME`: The target cluster name.
-   `ZONE`: The zone for your cluster (e.g., `us-central1-c`).
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"gs://<your_gcs_bucket>"`).
-   `WORKLOAD_IMAGE`: The Docker image for the workload. 
-   `WORKLOAD_NAME`: A unique name for your workload. 

## Run the recipe

### Configure environment settings

Before running any commands in this section, ensure you have set the environment
variables in `run_recipe.sh`. Don't forget to replace placeholders like `<YOUR_HF_TOKEN>`, `<YOUR_GCS_BUCKET>`, etc.

### Run muse-glimmer-30b Pretraining Workload

To run the benchmark, first make the script executable, edit it to configure
environment variables, and then run it:

```bash
chmod +x run_recipe.sh
nano run_recipe.sh
./run_recipe.sh
```

## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and logs.

### Follow Workload and View Metrics

List workloads using Cluster Toolkit (`gcluster job list`):

```bash
gcluster job list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

## Check results

After the job completes, you can check the results by accessing logs from the bucket `${BASE_OUTPUT_DIR}`.

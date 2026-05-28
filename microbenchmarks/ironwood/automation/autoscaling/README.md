# Ironwood Benchmark Automation With CCC for nodepool creation

This directory contains the automation framework for running TPU microbenchmarks (HBM, Host-Device, Collectives, etc.) on GKE clusters with autoscaling enabled through CCC. The tool simplifies the workflow of launching multiple benchmark jobs via [Kueue](https://kueue.sigs.k8s.io/), monitoring their status, handling retries, and aggregating the final results into a unified format.

The autoscaling version of the script uses CustomComputeClass (CCC) to manage the creation and deletion of the required nodepools automatically based on the workloads.

## Overview

The automation workflow consists of three main stages:
1.  **Launch**: Submits Kubernetes Jobs for various benchmark configurations (e.g., different topologies like 2x2x1, 2x2x2) using Kueue for queue management.
2.  **Monitor & Retry**: Watches the jobs until completion. If any job fails, it automatically retries them (up to 3 times by default).
3.  **Aggregate**: Once all jobs succeed, an aggregator job is launched to collect all intermediate results from GCS and consolidate them into summary TSV files.

## Prerequisites

Before running the automation script, ensure the following requirements are met:

### 1. Environment Setup
*   **GKE Cluster**: You must have a GKE cluster.
*   **Kubectl**: Ensure `kubectl` is installed and authenticated to your cluster.
*   **GCS Bucket**: A Google Cloud Storage bucket is required to store intermediate and final aggregated results.
    ```bash
    gcloud storage buckets create gs://my-unique-bucket-name --location=us-central1
    ```

### 2. Install Kueue
The automation relies on Kueue for job queuing. Check if it's already installed:

```bash
kubectl get namespace kueue-system
```

If you see `Error from server (NotFound)`, install it with:

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/manifests.yaml
```

### 3. Verify requirments for CCC
In order for CCC to work the correct set of CCC templates need to be created. If you have not already done so, allowing pre-flight checks to run
when the script prompts for it will install all the required CCC templates (templates for different TPU topologies 2x2x1, 2x2x2, etc)

## Directory Structure

*   `automation_launch.sh`: The main entry point script. Manages the full lifecycle of the benchmark run.
*   `check_ccc_resources.sh`: Validation script that makes sure all CCC related resources are created.
*   `create_ccc_templates.sh`: Create the required CCC related resources.
*   `../aggregator.py`: Python script that downloads results from GCS and produces summary tables.
*   `../aggregator.yaml`: Kubernetes Job definition for running the aggregator.
*   `job-queue-CCC.yaml`: Kueue resource definitions (ClusterQueue, LocalQueue).
*   `*.yaml`: Benchmark job configurations (e.g., `tpu7x-2x2x1-hbm.yaml`).

## Configuration

You can configure the behavior using the following environment variable:

| Variable | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `GCS_BUCKET_ROOT_DIR` | The root GCS path where results will be stored. Must start with `gs://`. | **Yes** | `gs://example-microbenchmark` (Change this!) |

## Usage Guide

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/google/accelerator-microbenchmarks.git
    cd accelerator-microbenchmarks
    # Switch to the correct branch if necessary
    git checkout tpu7x-auto
    ```

2.  **Set the GCS Bucket**:
    Export the path to your GCS bucket. This is where all results will be saved.
    ```bash
    export GCS_BUCKET_ROOT_DIR="gs://your-unique-bucket-name/benchmark_runs/$(date +%Y%m%d_%H%M%S)"
    ```

3.  **Run the Automation Script**:
    Execute the launch script from the root of the repository.
    ```bash
    bash Ironwood/guides/automation/automation_launch.sh
    ```

    **What happens next?**
    *   If pre-flight checks are enabled, will check and CCC resources (and create if needed) and check GCS permissions
    *   It applies the Kueue job queue.
    *   It submits the benchmark jobs defined in the script (e.g., HBM tests).
    *   It waits for jobs to finish, retrying any failures up to 3 times.
    *   Finally, it launches the `aggregator` job.

## Output

After the automation completes, check your GCS bucket (`GCS_BUCKET_ROOT_DIR`). You will find:

*   **`aggregated_results/`**: Contains the final summary CSV/TSV files (e.g., `hbm.tsv`, `collectives.tsv`).
*   **`<job-name>/`**: Directories for each individual job containing intermediate results.

## Troubleshooting

### Job Failures
If jobs fail even after retries:
1.  Check the script output to see which specific jobs failed.
2.  Inspect the logs of a failed job using `kubectl logs job/<job-name>`.
3.  Manually retry a specific job if needed using the command printed by the script at the end of the run.

### Missing Results
If the `aggregated_results` folder is empty:
1.  Check the logs of the aggregator job:
    ```bash
    kubectl logs job/aggregator
    ```
2.  Ensure the `GCS_BUCKET_ROOT_DIR` was accessible by the pods (check Workload Identity or service account permissions if running in a restricted project).

# Microbenchmarks
Microbenchmarks that assess the performance of individual operations and components on Ironwood accelerators with JAX.

## Prerequisites

- Ensure [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install) is installed in your local machine.
- Ensure `kubectl` is installed in your local machine: `gcloud components install kubectl
`
- Create a Cloud TPU GKE cluster, which uses GKE version >= `1.34.0-gke.2201000`
- Ensure you have sufficient number of chips. This guide provisions two slices; one `2x2x1` (single host) slice and one `4x4x4` (multi host) slice and therefore requires 68 chips.

Refer to ["Deploy TPU workloads in GKE" guide](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus) for more information about how to set up a TPU GKE cluster with sufficient resources.

## Setup

Create one `2x2x1` nodepool in your GKE cluster:

```bash
# The default GKE version may be lower than `1.34.0-gke.2201000`. Check if `node-version` should be specified before applying this command.
gcloud container node-pools create ${SINGLE_SLICE_NODE_POOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=tpu7x-standard-4t \
    --location=${LOCATION} \
    --node-locations=${LOCATION} \
    --project=${PROJECT_ID} \
    --num-nodes=1 \
    --reservation=${RESERVATION_NAME} \
    --reservation-affinity=specific
```

Create one `4x4x4` nodepool in your GKE cluster using workload policy:

```bash
gcloud compute resource-policies create workload-policy ${WORKLOAD_POLICY_NAME} \
    --type HIGH_THROUGHPUT \
    --accelerator-topology 4x4x4 \
    --project ${PROJECT_ID} \
    --region ${REGION}

# The default GKE version may be lower than `1.34.0-gke.2201000`. Check if `node-version` should be specified before applying this command.
gcloud container node-pools create ${MULTI_SLICE_NODE_POOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=tpu7x-standard-4t \
    --placement-policy=${WORKLOAD_POLICY_NAME} \
    --project ${PROJECT_ID} \
    --location=${LOCATION} \
    --reservation=${RESERVATION_NAME} \
    --reservation-affinity=specific
```

Note: `$LOCATION` can either be the GCP zone or region depending on the GKE cluster configuration.

Nodepool provisioning commands may vary depending on the GKE cluster setup. Please refer to ["Deploy TPU workloads in GKE" guide](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#create-node-pool) for all of the options.

If the default Google Kubernetes Engine (GKE) version is less than `1.34.0-gke.2201000`, the minimum required version for Ironwood, the node-version argument must be specified during node pool creation. Furthermore, the cluster's GKE version must also be equal to or greater than `1.34.0-gke.2201000`, settable via the cluster-version argument.


## Running the microbenchmarks in the GKE cluster

Get credentials for the GKE cluster:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --location ${LOCATION} --project ${PROJECT_ID}
```

Verify that you have sufficient `gke-tpu-.*` nodes:

```bash
kubectl get nodes
```

### Deploying a single host job

Create a job manifest to run `2x2x1` microbenchmarks (`tpu7x-2x2x1-micobenchmarks.yaml`):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tpu7x-single-host-microbenchmark
spec:
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: tpu7x
    cloud.google.com/gke-tpu-topology: 2x2x1
  containers:
  - name: tpu-job
    image: python:3.12
    ports:
    - containerPort: 8431
    securityContext:
      privileged: false
    command:
    - bash
    - -c
    - |
      set -ex

      git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git
      cd accelerator-microbenchmarks
      pip install -r requirements.txt

      sh ./Ironwood/scripts/run_training_compute_microbenchmark.sh

    resources:
      requests:
        google.com/tpu: 4
      limits:
        google.com/tpu: 4
```

Deploy the 2x2x1 microbenchmarks:

```bash
kubectl apply -f tpu7x-2x2x1-micobenchmarks.yaml
```

Monitor the results:

```bash
kubectl logs tpu7x-single-host-microbenchmark
```

Cleanup the job:

```bash
kubectl delete pod tpu7x-single-host-microbenchmark
```

### Deploying a multi host job

Create a job manifest to run `4x4x4` microbenchmarks (`tpu7x-4x4x4-micobenchmarks.yaml`):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: headless-svc
spec:
  clusterIP: None
  selector:
    job-name: tpu7x-multi-host-microbenchmark
---
apiVersion: batch/v1
kind: Job
metadata:
  name: tpu7x-multi-host-microbenchmark
spec:
  completionMode: Indexed
  parallelism: 16
  completions: 16
  backoffLimit: 0
  template:
    spec:
      subdomain: headless-svc
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 4x4x4
      containers:
      - name: jax-tpu
        image: python:3.12
        securityContext:
          privileged: false
        env:
        - name: JAX_PLATFORMS
          value: "tpu,cpu"
        - name: TPU_VMODULE
          value: "singleton_tpu_system_manager=10,tpu_version_flag=10,device_util=10,device_scanner=10,mesh_builder=10,master=10"
        - name: XLA_IR_DEBUG
          value: "1"
        - name: XLA_HLO_DEBUG
          value: "1"
        command:
        - bash
        - -c
        - |
          set -ex

          git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git
          cd accelerator-microbenchmarks
          pip install -r requirements.txt

          sh ./Ironwood/scripts/run_ici_microbenchmark_full.sh

        resources:
          requests:
            google.com/tpu: 4
          limits:
            google.com/tpu: 4
```

Deploy the 4x4x4 microbenchmarks:

```bash
kubectl apply -f tpu7x-4x4x4-micobenchmarks.yaml
```

List all the jobs in the pod to get the name of a job:
```bash
kubectl get pods
```

Monitor the results of the job name picked above (monitoring only a single job is sufficient as all jobs will have the same logs):

```bash
kubectl logs tpu7x-multi-host-microbenchmark-0-XXXXX
```

Cleanup the job:

```bash
kubectl delete -f tpu7x-4x4x4-micobenchmarks.yaml
```

## Microbenchmark scripts

### Compute

`Ironwood/scripts/run_training_compute_microbenchmark.sh` script runs the training compute microbenchmark, including gemm, add, quantization, and transpose quantization:

| Operation | Function Description | Formula / Logic |
| :--- | :--- | :--- |
| **`gemm_multiple_run`** | **Configurable GEMM.** Benchmarks matmul with configurable input types (supports `float8_e4m3fn`, `bfloat16`). Accumulation is FP32, output cast to BF16. Rerun multiple times and record all profilers.<br>**Example YAML.** [Ironwood/configs/training/gemm_multiple_run_more.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/gemm_multiple_run_more.yaml) | $O_{bf16} = (A \times B)$ |
| **`gemm_simple`** | **Basic FP8 GEMM.** Benchmarks pure FP8 matmul with FP32 accumulation.<br>**Example YAML.** [Ironwood/configs/training/gemm_simple.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/gemm_simple.yaml) | $O_{bf16} = (A_{fp8} \times B_{fp8})$ |
| **`gemm`** | **FP8 GEMM + Scaling.** Performs FP8 matmul and applies scaling factors (dequantization) to the result.<br>**Example YAML.** [Ironwood/configs/training/gemm.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/gemm.yaml) | $O_{bf16} = (A_{fp8} \times B_{fp8}) \times (S_A \times S_B^T)$ |
| **`gemm_accum`** | **FP8 GEMM + Accumulation.** Performs FP8 matmul with scaling factors and accumulates the result into an existing FP32 output buffer.<br>**Example YAML.** [Ironwood/configs/training/gemm_accum.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/gemm_accum.yaml) | $O_{fp32} \ += (A_{fp8} \times B_{fp8}) \times (S_A \times S_B^T)$ |
| **`gemm_fp8_rowwise`** | **FP8 Row-wise GEMM.** Quantizes inputs dynamically (row-wise/channel-wise) using absmax calibration before performing the matrix multiplication.<br>**Example YAML.** [Ironwood/configs/training/gemm_fp8_rowwise.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/gemm_fp8_rowwise.yaml) | $O_{bf16} = (Quant(A) \times Quant(B))$ |
| **`add`** | **Element-wise Addition.** Adds two BF16 tensors.<br>**Example YAML.** [Ironwood/configs/training/add.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/add.yaml) | $Z = X + Y$ |
| **`quantization`** | **Dynamic Quantization.** Quantizes a BF16 input tensor to FP8 using dynamic scaling (absmax calibration). Returns quantized values and scale factors.<br>**Example YAML.** [Ironwood/configs/training/quantization.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/quantization.yaml) | $S = \frac{Max}{absmax(X)}$, $O = Cast(\frac{X}{S})$ |
| **`transpose_quantization`** | **Transpose + Quantization.** Transposes a BF16 input tensor first, then applies dynamic quantization.<br>**Example YAML.** [Ironwood/configs/training/transpose_quantization.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/configs/training/transpose_quantization.yaml) | $S = \frac{Max}{absmax(X^T)}$, $O = Cast(\frac{X^T}{S})$ |

### Collectives

`Ironwood/scripts/run_ici_microbenchmark_full.sh` script runs the collective microbenchmarks, including psum, psum_scatter, all_gather, and all_to_all:

| Operation | Function Description | Formula / Logic |
| :--- | :--- | :--- |
| **`psum`** | **All-Reduce (Sum).** Sums tensors across devices. | $O = \sum X_i$ |
| **`psum_scatter`** | **Reduce-Scatter (Sum).** Sums tensors and scatters results. | $O_i = \sum X_i$ (scattered) |
| **`all_gather`** | **All-Gather.** Gathers tensors from all devices. | $O = [X_0, X_1, ...]$ |
| **`all_to_all`** | **All-to-All.** Scatters and gathers data between all devices. | Scatters/Gathers across devices |


## Results

### GEMM
The table below summarizes the throughput performance (in `TFLOP/s` per device) for gemm_multiple_run across varying bfloat16 matrix sizes. (Used config: gemm_multiple_run_more.yaml.)

| Matrix Size (m=n=k) | Throughput (`TFLOP/s/device`) |
| :--- | :--- |
| 128 | 2.589961271 |
| 256 | 18.28645367 |
| 512 | 105.0783466 |
| 1024 | 349.7270639 |
| 2048 | 679.284728 |
| 4096 | 892.9445127 |
| 16384 | 950.1286983 |
| 32768 | 956.3824592 |


## Examine the outputs

The benchmarks will print metrics to the terminal. If you wish to dump formatted metrics in a file, you may set this parameter in your YAML file:
* `csv_path`: Dumps the benchmark metrics in a CSV.
Examples can be found in the YAML files under config/ directory.

If you wish to generate the xprof profile, set this parameter in the YAML file:
* `trace_dir`: Dumps the xprof profile to either a local location or GCS bucket.
Examples can be found in the YAML files under config/ directory.
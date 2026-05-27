# Inference Attention Microbenchmarks
This microbenchmark captures the performance of the Ragged-Paged Attention (RPAv3) kernel, with proper tuning.

## Prerequisites and Setup

Please follow the instructions located [here](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/Ironwood_Microbenchmarks_readme.md#prerequisites) to set up GKE and [here](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/main/Ironwood/Ironwood_Microbenchmarks_readme.md#setup) to setup your specific GKE resources.

## Running the Infernece Attention Microbenchmarks in a GKE cluster

Get credentials for the GKE cluster:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --location ${LOCATION} --project ${PROJECT_ID}
```

Verify that you have sufficient `gke-tpu-.*` nodes:

```bash
kubectl get nodes
```

### Deploying a single host job

Create a job manifest to run `2x2x1` microbenchmarks (`tpu7x-2x2x1-inference-attention-microbenchmark.yaml`):

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
    image: vllm/vllm-tpu:ironwood
    ports:
    - containerPort: 8431
    securityContext:
      privileged: false
    command:
    - bash
    - -c
    - |
      set -ex
      git clone https://github.com/vllm-project/tpu-inference.git
      cd tpu-inference
      git checkout jacobplatin/janus-benchmarking-ext
      pip install tensorflow
      python benchmarking/benchmark_janus.py
    resources:
      requests:
        google.com/tpu: 4
      limits:
        google.com/tpu: 4
```

> **_NOTE:_** the `python benchmarking/benchmark_janus.py` accepts the following flags:
>
> * `is_prefill`: Set to true to run prefill configs (defauls to `False`).
> * `outdir`: Directory to save results and profiles (defaults to `benchmark-results`).

Deploy the 2x2x1 inference attention microbenchmarks:

```bash
kubectl apply -f tpu7x-2x2x1-inference-attention-microbenchmark.yaml
```

Monitor the results:

```bash
kubectl logs -f tpu7x-single-host-microbenchmark
```

Cleanup the job:

```bash
kubectl delete pod tpu7x-single-host-microbenchmark
```

## Examine the outputs

The benchmarks will print metrics to the terminal.  In addition, the traces and a CSV with all results will be saved to `outdir` (see above).

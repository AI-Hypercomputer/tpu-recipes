# Serve Qwen3.5 397B with vLLM on Ironwood TPU

In this guide, we show how to serve Qwen3.5 397B models (e.g., `Qwen/Qwen3.5-397B-A17B-FP8`) with vLLM on Ironwood (TPU v7x) using GKE.

## Verified Models

The following larger Qwen 3.5 models are verified for deployment on TPU.

### Verified Models

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :---- | :---- | :---- | :---- |
| Qwen 3.5 397B FP8 | 397B (17B active) | 8× | [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) |

## Cluster Prerequisites

Before deploying the vLLM workload, ensure your GKE cluster is configured with the necessary networking and identity features.

### Define parameters

```bash
# Set variables
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=<YOUR_REGION>
export ZONE=<YOUR_ZONE>
export NODEPOOL_NAME=<YOUR_NODEPOOL_NAME>
```

### Create nodepool

Create a TPU v7 (Ironwood) nodepool. Note that running this model natively requires 8 chips. For `tpu7x` this is achieved using 2 nodes in a 2x2x1 topology.

```bash
gcloud container node-pools create ${NODEPOOL_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --num-nodes=2 \
  --machine-type=tpu7x-standard-4t \
  --cluster=${CLUSTER_NAME}
```

## Deploy vLLM Workload on GKE

1. Configure kubectl to communicate with your cluster

    ```bash
    gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${ZONE}
    ```

2. Create a Kubernetes Secret for Hugging Face credentials

    ```bash
    export HF_TOKEN=YOUR_TOKEN
    kubectl create secret generic hf-secret \
        --from-literal=hf_api_token=${HF_TOKEN}
    ```

3. Install LeaderWorkerSet

    Because this recipe requires multi-host inference (spanning 2 nodes), you must install LeaderWorkerSet in your cluster if it isn't already installed:

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/lws/releases/download/v0.3.0/install.yaml
    ```

4. Apply the vLLM manifest using the provided `vllm-tpu.yaml` file in this directory:

    ```bash
    kubectl apply -f vllm-tpu.yaml
    ```

5. Check the status of the server

    To see the deployment status and monitor the pod as it starts up, run:
    
    ```bash
    kubectl get pods -w
    ```
    
    Wait for the leader pod to reach `Running` state. Once running, you can follow the server startup logs:
    
    ```bash
    kubectl logs vllm-tpu-server-0 -f
    ```

    At the end of the server startup you'll see logs such as:

    ```bash
    ...
    (APIServer pid=1) INFO:     Started server process [1]
    (APIServer pid=1) INFO:     Waiting for application startup.
    (APIServer pid=1) INFO:     Application startup complete.
    ```

6. Interact with the model using curl

    Once the application startup is complete, run:

    ```bash
    kubectl port-forward service/vllm-service 8000:8000

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3.5-397B-A17B-FP8",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0,
            "top_p": 1.0
        }'
    ```

### (Optional) Benchmark via Service

To benchmark the server, we use the InferenceX client from SemiAnalysisAI.
Reference: <https://github.com/SemiAnalysisAI/InferenceX>.

First, download the client code locally if you want to inspect it: `git clone https://github.com/SemiAnalysisAI/InferenceX.git`

1. Execute a short benchmark against the server using one of the following workloads.

    #### Workload 1k/8k

    Save the following manifest as `vllm-benchmark-1k8k.yaml` and apply it using `kubectl apply -f vllm-benchmark-1k8k.yaml`.

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench-1k8k
      namespace: default
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:nightly-20260616-df47e95-a30addc
        command: ["/bin/bash", "-c"]
        args:
        - |
          while ! curl http://vllm-service:8000/ping; do sleep 30 && echo 'Waiting for server...'; done
          apt-get update && apt-get install -y git && \
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /ubench/inferencex && \
          cd /ubench/inferencex && \
          git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
          python3 /ubench/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --dataset-name=random \
            --random-input-len=1024 \
            --random-output-len=8192 \
            --random-range-ratio=0.8 \
            --host=vllm-service \
            --port=8000 \
            --model=Qwen/Qwen3.5-397B-A17B-FP8 \
            --tokenizer=Qwen/Qwen3.5-397B-A17B-FP8 \
            --ignore-eos \
            --num-prompts=640 \
            --max-concurrency=64
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
    ```

    #### Workload 8k/1k

    Save the following manifest as `vllm-benchmark-8k1k.yaml` and apply it using `kubectl apply -f vllm-benchmark-8k1k.yaml`.

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench-8k1k
      namespace: default
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:nightly-20260616-df47e95-a30addc
        command: ["/bin/bash", "-c"]
        args:
        - |
          while ! curl http://vllm-service:8000/ping; do sleep 30 && echo 'Waiting for server...'; done
          apt-get update && apt-get install -y git && \
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /ubench/inferencex && \
          cd /ubench/inferencex && \
          git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
          python3 /ubench/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --dataset-name=random \
            --random-input-len=8192 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --host=vllm-service \
            --port=8000 \
            --model=Qwen/Qwen3.5-397B-A17B-FP8 \
            --tokenizer=Qwen/Qwen3.5-397B-A17B-FP8 \
            --ignore-eos \
            --num-prompts=640 \
            --max-concurrency=64
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
    ```

2. Check the progress of benchmark:

    ```bash
    kubectl logs -f vllm-bench-1k8k # For 1k/8k workload
    ```

    Example Output:
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     640
    Benchmark duration (s):                  xx.xx
    Total input tokens:                      xxxx
    Total generated tokens:                  xxxx

    Request throughput (req/s):              xx.xx
    Output token throughput (tok/s):         xxxx.xx
    Total Token throughput (tok/s):          xxxx.xx

    ---------------Time to First Token----------------
    Mean TTFT (ms):                          xxxx.xx
    Median TTFT (ms):                        xxxx.xx
    P99 TTFT (ms):                           xxxx.xx

    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          xx.xx
    Median TPOT (ms):                        xx.xx
    P99 TPOT (ms):                           xx.xx

    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           xx.xx
    Median ITL (ms):                         xx.xx
    P99 ITL (ms):                            xx.xx
    ==================================================
    ```

    Workload (input tokens/output tokens) | Output Token Throughput (tok/s) Per Chip
    :------- | :---------------------------------------
    1k/8k    | 2436.35 tok/s (609.09 tok/s/chip)
    8k/1k    | 1513.05 tok/s (378.26 tok/s/chip)

    **Note**: These benchmark results are based on the `InferenceX` client. The
    development team is continuously improving and optimizing performance; as such,
    these results are subject to change, and improved or optimized figures may be
    published in the future.

3. Clean up

    ```bash
    kubectl delete -f vllm-benchmark-1k8k.yaml
    kubectl delete -f vllm-benchmark-8k1k.yaml
    kubectl delete -f vllm-tpu.yaml
    ```

# Serve Gemma 4 IT with vLLM on Ironwood TPU

In this guide, we show how to serve Gemma 4 IT models (e.g., `google/gemma-4-31B-it`) with vLLM on Ironwood (TPU v7x) using GKE.

> Note: These setup instructions are **specifically for Gemma 4** on TPU and may not work for other models as it uses custom wheels and source builds for vllm and transformers.

## Verified Models

The following larger Gemma 4 models are verified for deployment on TPU.

### Verified Models

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :---- | :---- | :---- | :---- |
| Gemma 4 31B IT | 31B | 1× | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 26B-A4B IT (MoE) | 26B (4B active) | 1× | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |

> [!NOTE]
> Gemma 4 E2B IT, Gemma 4 E4B IT are currently not verified for TPU deployment.

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

Create a TPU v7 (Ironwood) nodepool.

```bash
gcloud container node-pools create ${NODEPOOL_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --num-nodes=1 \
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
        --namespace=gemma4-test \
        --from-literal=hf_api_token=${HF_TOKEN}
    ```

3. Apply the vLLM manifest using the provided `gemma4-server.yaml` file in this directory:

> [!TIP]
> If you are using pre-built images optimized for Gemma 4 (which have the required custom wheels baked in), the environment variables for vision stability (`VLLM_WORKER_MULTIPROC_METHOD=fork`, etc.) are already set inside the image. If you are using a standard image, you must define them in the `env:` section of the container spec.
`

4. Apply the vLLM manifest

    ```bash
    kubectl apply -f gemma4-server.yaml
    ```

5. Interact with the model using curl

    ```bash
    kubectl port-forward service/vllm-service 8000:8000

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "google/gemma-4-31B-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?auto=format&fit=crop&w=800&q=80"
                            }
                        },
                        {
                            "type": "text", 
                            "text": "Describe the image above"
                        }
                    ]
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

First, download the client code: `git clone https://github.com/SemiAnalysisAI/InferenceX.git`

1. Execute a short benchmark against the server using one of the following workloads.

    #### Workload 1k/1k

    Save the following manifest as `vllm-benchmark-1k1k.yaml` and apply it using `kubectl apply -f vllm-benchmark-1k1k.yaml`.

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench-1k1k
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:gemma4
        command: ["/bin/bash", "-c"]
        args:
        - |
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /ubench/inferencex && \
          cd /ubench/inferencex && \
          git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
          python3 /ubench/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-service \
            --port=8000 \
            --model=google/gemma-4-31B-it \
            --tokenizer=google/gemma-4-31B-it \
            --dataset-name=random \
            --random-input-len=1024 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --num-prompts=320 \
            --max-concurrency=64 \
            --ignore-eos
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
    ```

    #### Workload 1k/8k

    Save the following manifest as `vllm-benchmark-1k8k.yaml` and apply it using `kubectl apply -f vllm-benchmark-1k8k.yaml`.

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench-1k8k
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:gemma4
        command: ["/bin/bash", "-c"]
        args:
        - |
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /ubench/inferencex && \
          cd /ubench/inferencex && \
          git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
          python3 /ubench/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-service \
            --port=8000 \
            --model=google/gemma-4-31B-it \
            --tokenizer=google/gemma-4-31B-it \
            --dataset-name=random \
            --random-input-len=1024 \
            --random-output-len=8192 \
            --random-range-ratio=0.8 \
            --num-prompts=320 \
            --max-concurrency=64 \
            --ignore-eos
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
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:gemma4
        command: ["/bin/bash", "-c"]
        args:
        - |
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /ubench/inferencex && \
          cd /ubench/inferencex && \
          git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
          python3 /ubench/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-service \
            --port=8000 \
            --model=google/gemma-4-31B-it \
            --tokenizer=google/gemma-4-31B-it \
            --dataset-name=random \
            --random-input-len=8192 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --num-prompts=320 \
            --max-concurrency=64 \
            --ignore-eos
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
    ```

2. Check the progress of benchmark:

    ```bash
    kubectl logs -f vllm-bench-1k1k # For 1k/1k workload
    ```

    Example Output:
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     320
    Failed requests:                         0
    Benchmark duration (s):                  xx
    ...
    ```

3. Clean up

    ```bash
    kubectl delete -f vllm-benchmark-1k1k.yaml
    kubectl delete -f vllm-benchmark-1k8k.yaml
    kubectl delete -f vllm-benchmark-8k1k.yaml
    ```


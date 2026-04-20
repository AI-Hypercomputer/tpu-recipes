# Serve Qwen3-Coder-480B-A35B-Instruct-FP8 with vLLM on Ironwood TPU

In this guide, we show how to serve
[Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) on Ironwood (TPU7x).

## Install `gcloud cli`

You can reproduce this experiment from your dev environment (e.g. your laptop).
You need to install `gcloud` locally to complete this tutorial.

To install `gcloud cli` please follow this guide:
[Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

Once it is installed, you can login to GCP from your terminal with this command:
`gcloud auth login`.

## Cluster Prerequisites

Before deploying the vLLM workload, ensure your GKE cluster is configured with
the necessary networking and identity features.

### Define parameters

  ```bash
  # Set variables if not already set
  export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
  export PROJECT_ID=<YOUR_PROJECT_ID>
  export REGION=<YOUR_REGION>
  export ZONE=<YOUR_ZONE> # e.g., us-central1-a
  export NODEPOOL_NAME=<YOUR_NODEPOOL_NAME>
  export RESERVATION_NAME=<YOUR_RESERVATION_NAME> # Optional, if you have a reservation
  ```

### Create new cluster

Note: If a cluster already exists follows the steps in next section

The command below creates a cluster with the basic features required for this
recipe. For a more general guide on cluster creation please follow the GKE
documentation.

* [Creating Standard regional cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/creating-a-regional-cluster)
* [Creating Autopilot cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/creating-an-autopilot-cluster)

1. **Create GKE cluster with all the required features enabled**.

  ```bash
  gcloud container clusters create $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --location=$REGION \
    --workload-pool=$PROJECT_ID.svc.id.goog \
    --release-channel=rapid \
    --num-nodes=1 \
    --gateway-api=standard \
    --addons GcsFuseCsiDriver,HttpLoadBalancing
  ```

### Updating existing cluster

Check if the following features are enabled in the cluster, if not use the
following steps to enable the required features.

1. **Enable Workload Identity:**. The cluster and the nodepool needs to have
    workload identity enabled.

    To enable in cluster:

    ```bash
    gcloud container clusters update ${CLUSTER_NAME} \
      --location=${REGION} \
      --workload-pool=$PROJECT_ID.svc.id.goog
    ```

    To enable in existing nodepool.

    ```bash
    gcloud container node-pools update ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --workload-metadata=GKE_METADATA
    ```

2. **Enable HTTP Load Balancing:** The cluster must have the
    `HttpLoadBalancing` add-on enabled. This is typically enabled by default,
    but you can confirm or add it:

    ```bash
    gcloud container clusters update ${CLUSTER_NAME} \
      --region ${REGION} \
      --project ${PROJECT_ID} \
      --update-addons=HttpLoadBalancing=ENABLED
    ```

3. **Enable Gateway API:** Enable the Gateway API on the cluster to allow GKE
    to manage Gateway resources. Note: This step is optional and needed only if
    inference gateway will be used. More details about the inference gateway can
    be found at
    [GKE Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway).

    ```bash
    gcloud container clusters update ${CLUSTER_NAME} \
      --region ${REGION} \
      --project ${PROJECT_ID} \
      --gateway-api=standard
    ```

### Create proxy-only subnet

1. **Create Proxy-Only Subnet:** A proxy-only subnet is required for the
    regional managed proxy.

    ```bash
    # Set variables if not already set
    export PROXY_ONLY_SUBNET_NAME="tpu-proxy-subnet"
    export VPC_NETWORK_NAME="default" # Or your custom VPC
    export PROXY_ONLY_SUBNET_RANGE="10.129.0.0/23" # Example range

    gcloud compute networks subnets create ${PROXY_ONLY_SUBNET_NAME} \
      --purpose=REGIONAL_MANAGED_PROXY \
      --role=ACTIVE \
      --region ${REGION} \
      --network "${VPC_NETWORK_NAME}" \
      --range "${PROXY_ONLY_SUBNET_RANGE}" \
      --project ${PROJECT_ID}
    ```

### Create nodepool (single host)

1. **Create TPU v7 (Ironwood) nodepool**. If a node pool does not already exist
create a node pool with a **single TPU v7 node** in 2x2x1 configuration.

    ```bash
    gcloud container node-pools create ${NODEPOOL_NAME} \
      --project=${PROJECT_ID} \
      --location=${REGION} \
      --node-locations=${ZONE} \
      --num-nodes=1 \
      --reservation=${RESERVATION_NAME} \
      --reservation-affinity=specific \
      --machine-type=tpu7x-standard-4t \
      --cluster=${CLUSTER_NAME}
    ```

## Deploy vLLM Workload on GKE

1. Configure kubectl to communicate with your cluster

    ```bash
    gcloud container clusters get-credentials ${CLUSTER_NAME} --location=us-central1-c
    ```

2. Generate a new
    [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) if you
    don't already have one (NOTE: ensure that you have access to the model on
    Hugging Face):

    1. Go to **Your Profile > Settings > Access Tokens**.
    2. Select **Create new token**.
    3. Specify a name of your choice and a role with at least **Read**
        permissions.
    4. Select **Generate a token**.

3. Create a Kubernetes Secret for Hugging Face credentials

    ```bash
    export HF_TOKEN=YOUR_TOKEN
    kubectl create secret generic hf-secret \
        --from-literal=hf_api_token=${HF_TOKEN}
    ```

4. Save this yaml file as `vllm-tpu.yaml`

    **Note:** This config is for 1k/8k and 8k/1k workloads. If users want to run
    a 1k/1k (other) workload, please change `max-model-len` and restart the
    server.

    ```
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
      name: hyperdisk-balanced-tpu
    provisioner: pd.csi.storage.gke.io
    parameters:
      type: hyperdisk-balanced
    reclaimPolicy: Delete
    volumeBindingMode: WaitForFirstConsumer
    allowVolumeExpansion: true
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: hd-claim
    spec:
      storageClassName: hyperdisk-balanced-tpu
      accessModes:
        - ReadWriteOnce
      resources:
        requests:
          storage: 700Gi
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-tpu
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: vllm-tpu
      template:
        metadata:
          labels:
            app: vllm-tpu
        spec:
          nodeSelector:
            cloud.google.com/gke-tpu-accelerator: tpu7x
            cloud.google.com/gke-tpu-topology: 2x2x1
          containers:
          - name: vllm-tpu
            image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626
            command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
            args:
            - --host=0.0.0.0
            - --port=8000
            - --seed=42
            - --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
            - --tensor-parallel-size=8
            - --data-parallel-size=1
            - --max-model-len=10240
            - --max-num-batched-tokens=8192
            - --max-num-seqs=512
            - --gpu-memory-utilization=0.8
            - --block-size=256
            - --kv-cache-dtype=fp8
            - --enable-expert-parallel
            - --no-enable-prefix-caching
            - --async-scheduling
            - --download-dir=/data
            env:
            - name: HF_HOME
              value: /data
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: hf_api_token
            - name: MODEL_IMPL_TYPE
              value: vllm
            - name: USE_BATCHED_RPA_KERNEL
              value: "1"
            ports:
            - containerPort: 8000
            resources:
              limits:
                google.com/tpu: '4'
              requests:
                google.com/tpu: '4'
            readinessProbe:
              tcpSocket:
                port: 8000
              initialDelaySeconds: 15
              periodSeconds: 10
            volumeMounts:
            - mountPath: "/data"
              name: data-volume
            - mountPath: /dev/shm
              name: dshm
          volumes:
          - emptyDir:
              medium: Memory
            name: dshm
          - name: data-volume
            persistentVolumeClaim:
              claimName: hd-claim
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-service
    spec:
      selector:
        app: vllm-tpu
      type: LoadBalancer
      ports:
        - name: http
          protocol: TCP
          port: 8000
          targetPort: 8000
    ```

5. Apply the vLLM manifest by running the following command

    ```bash
    kubectl apply -f vllm-tpu.yaml
    ```

    At the end of the server startup you’ll see logs such as:

    ```
    $ kubectl logs deployment/vllm-tpu -f
    …
    …
    (APIServer pid=1) INFO:     Started server process [1]
    (APIServer pid=1) INFO:     Waiting for application startup.
    (APIServer pid=1) INFO:     Application startup complete.
    ```

6. Serve the model by port-forwarding the service

    ```bash
    kubectl port-forward service/vllm-service 8000:8000
    ```

7. Interact with the model using curl (from your workstation/laptop)

    ```bash
    curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
    ```

### End to End performance with smart routing

The `vllm-tpu.yaml` file you deployed creates a standard Kubernetes `Service` of
type `LoadBalancer`. However, to get the most performance out of the hardware
and measure maximum throughput, you also need to consider smart routing. For
this, you can use the **GKE Inference Gateway**. This provides a "smart" load
balancer optimized for LLMs with features including:

* **Precise Prefix-Cache Aware Routing:** The `InferencePool` resource enables
    intelligent routing that considers the KV cache state on the TPU/GPU
    ("KV-cache-awareness"). This is critical for prefix-heavy workloads (like
    chatbots), where it improves time-to-first-token (TTFT) latency by up to 96%
    at peak throughput by routing requests with the same prefix to the same
    accelerator.
* **Prefill/Decode Disaggregation:** This feature (called "Disaggregated
    Serving") separates the initial prompt processing (prefill) from token
    generation (decode). This improves throughput by 60% by allowing prefill and
    decode nodes to be scaled independently, increasing resource utilization.
* **Advanced Load Balancing:** The gateway is "load-aware" and offers
    experimental features like **predicted latency balancing** to prevent
    overloading replicas and improve P90 latency.
* **Customizable Scheduling & Prioritization:** The gateway allows for
    "SLA-awareness" and "customizable scheduling policies" to define which
    requests are more critical and process them first.

For more details on these performance numbers, see the Google Cloud blog
post:[Scaling high-performance inference cost-effectively](https://cloud.google.com/blog/products/ai-machine-learning/gke-inference-gateway-and-quickstart-are-ga).

1. Install inference objectives CRD

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.0.0/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml
    ```

2. Create the Gateway Resource

    This manifest creates an internal Gateway, which will provision an internal
    load balancer to act as the entry point for your inference requests.

    Save the following as `internal-gateway.yaml`:

    ```yaml
    apiVersion: gateway.networking.k8s.io/v1
    kind: Gateway
    metadata:
      name: internal-gateway
    spec:
      gatewayClassName: gke-l7-rilb
      listeners:
        - protocol: HTTP
          port: 80
          name: http
    ```

    Apply the manifest: `kubectl apply -f internal-gateway.yaml`

3. Create the InferencePool

    This resource groups your vLLM pods (identified by the `app: vllm-tpu` label
    from `vllm-tpu.yaml`) and enables advanced, cache-aware routing.

    ```bash
    helm install vllm-tpu-pool \
      --set inferencePool.modelServers.matchLabels.app=vllm-tpu \
      --set provider.name=gke \
      --set inferenceExtension.monitoring.gke.enabled=true \
      --version v1.1.0 \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
    ```

4. Configure the HTTPRoute

    This manifest links your `internal-gateway` to your `vllm-tpu-pool`. It
    routes all traffic from the gateway's `/v1/completions` path directly to the
    `InferencePool`.

    Save the following as `vllm-http-route.yaml`:

    ```yaml
    apiVersion: gateway.networking.k8s.io/v1
    kind: HTTPRoute
    metadata:
      name: vllm-http-route
    spec:
      parentRefs:
      - name: internal-gateway
      rules:
      - matches:
        - path:
            type: PathPrefix
            value: /v1/completions
        backendRefs:
        - name: vllm-tpu-pool
          group: "inference.networking.k8s.io"
          kind: InferencePool
    ```

    Apply the manifest: `kubectl apply -f vllm-http-route.yaml`

5. Validate the Inference Gateway

    You must run these commands from a GCE VM inside the same VPC as your GKE
    cluster, or a long running pod in the same cluster.

    > **Tip:** If you don't have a VM or existing pod, you can create a
    > temporary debug pod:
    >
    > ```bash
    > kubectl run debug-pod --image=curlimages/curl --restart=Never -- /bin/sh -c "sleep 3600"
    > ```
    >
    > Then execute commands from inside it:
    >
    > ```bash
    > kubectl exec -it debug-pod -- /bin/sh
    > ```

    Get the Internal Gateway IP (this may take a few minutes to become
    available):

    ```bash
    export GW_IP=""
    echo "Waiting for Gateway IP..."
    while [ -z "$GW_IP" ]; do
      GW_IP=$(kubectl get gateway internal-gateway -n default -o jsonpath='{.status.addresses[0].value}')
      [ -z "$GW_IP" ] && sleep 10
    done
    echo "Gateway Internal IP: $GW_IP"
    ```

    From your test VM, use curl to send a request to the Gateway IP.

    ```bash
    curl -X POST http://${GW_IP}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
      "prompt": "What is Qwen3-Coder 480B?",
      "max_tokens": 50
    }'
    ```

    **Expected Result:** You will receive a successful JSON response (e.g.,
    `HTTP/1.1 200 OK`) from the vLLM server, proving the request was correctly
    routed through the `internal-gateway` and `InferencePool`.

### (Optional) Benchmark via Service

To benchmark the server, we use the InferenceX client from SemiAnalysisAI.
Reference: <https://github.com/SemiAnalysisAI/InferenceX>.

First, download the client code: `git clone https://github.com/SemiAnalysisAI/InferenceX.git`

1. Execute a short benchmark against the server using one of the following
    workloads.

    #### Workload 1k/1k

    Save the following manifest as `vllm-benchmark-1k1k.yaml` and apply it using
    `kubectl apply -f vllm-benchmark-1k1k.yaml`.

    ```
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench-1k1k
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626
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
            --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
            --tokenizer=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
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

    Save the following manifest as `vllm-benchmark-1k8k.yaml` and apply it using
    `kubectl apply -f vllm-benchmark-1k8k.yaml`.

    ```
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench-1k8k
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626
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
            --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
            --tokenizer=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
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

    Save the following manifest as `vllm-benchmark-8k1k.yaml` and apply it using
    `kubectl apply -f vllm-benchmark-8k1k.yaml`.

    ```
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench-8k1k
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-bench
        image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626
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
            --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
            --tokenizer=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
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

    ```
    $ kubectl logs -f vllm-bench-1k1k # For 1k/1k workload
    …
    …
    ============ Serving Benchmark Result ============
    Successful requests:                     320
    Failed requests:                         0
    Benchmark duration (s):                  xx
    Total input tokens:                      xxx
    Total generated tokens:                  xxx
    Request throughput (req/s):              xx
    Output token throughput (tok/s):         xxx
    Peak output token throughput (tok/s):    xxx
    Peak concurrent requests:                64.00
    Total Token throughput (tok/s):          xxx
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          xxx
    Median TTFT (ms):                        xxx
    P99 TTFT (ms):                           xxx
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          xxx
    Median TPOT (ms):                        xxx
    P99 TPOT (ms):                           xxx
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           xxx
    Median ITL (ms):                         xxx
    P99 ITL (ms):                            xxx
    ==================================================
    ```

    Workload (input tokens/output tokens) | Output Token Throughput (tok/s) Per Chip
    :------- | :---------------------------------------
    1k/1k    | 1919.04 tok/s (479.76 tok/s/chip)
    1k/8k    | 2002.62 tok/s (500.66 tok/s/chip)
    8k/1k    | 967.98 tok/s (242.00 tok/s/chip)

    **Note**: These benchmark results are based on the `InferenceX` client. The
    development team is continuously improving and optimizing performance; as such,
    these results are subject to change, and improved or optimized figures may be
    published in the future.

3. Clean up

    ```
    kubectl delete -f vllm-benchmark-1k1k.yaml
    kubectl delete -f vllm-benchmark-1k8k.yaml
    kubectl delete -f vllm-benchmark-8k1k.yaml
    kubectl delete -f vllm-tpu.yaml
    ```

# Serve Kimi-K3 with vLLM (TorchTPU) Megakernels on Ironwood TPU (Disaggregated 1P1D)

In this guide, we show how to serve Kimi-K3 (`moonshotai/Kimi-K3`) with vLLM using the **TorchTPU Pallas megakernel** backend on Ironwood (TPU v7x) using GKE.

We deploy it as a disaggregated prefill/decode pair (**1P1D**) across two multi-host **tpu7x-standard-4t** node pools (`2x2x4` topology each: 8 hosts, 32 TPU v7x chips, 64 TensorCores in total) using Kubernetes **LeaderWorkerSet**, with **Tensor Parallelism (TP) size of 32** per role, **Raiden** KV transfer between the roles, **vllm-router** in front of the pair, and **DSpark** speculative decoding (`Inferact/Kimi-K3-DSpark`) on the decode role.

## Prerequisites

1. A GKE cluster with TPU v7x support (GKE version `1.34.0-gke.2201000` or later) with the LeaderWorkerSet (LWS) API, the Cloud Storage FUSE CSI driver, and Workload Identity enabled, plus at least one CPU node for the router and the benchmark client.
2. `gcloud`, `kubectl`, and `docker` (with BuildKit) installed and configured.
3. A CPU machine with ~3.5 TiB of local disk for the one-time weight pre-sharding (Step 4).

---

## Step 1: Define Parameters

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export IMAGE_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/<YOUR_AR_REPO>"
export IMAGE_URI="${IMAGE_REGISTRY}/vllm-tpu:k3-megakernel"
export GCS_MODEL_BUCKET="<YOUR_GCS_MODEL_BUCKET>"
export PREFILL_NODEPOOL="kimi-k3-prefill-pool"
export DECODE_NODEPOOL="kimi-k3-decode-pool"
export NAMESPACE="vllm-kimi-k3-mk"   # the namespace used in the manifests
```

> [!NOTE]
> Keep the image tag `vllm-tpu:k3-megakernel`; the manifests reference it as `<REGISTRY>/vllm-tpu:k3-megakernel`.

---

## Step 2: Create the TPU Node Pools

Create two `2x2x4` node pools, one for prefill and one for decode (4 nodes with 4 chips each per pool):

```shell
for POOL in ${PREFILL_NODEPOOL} ${DECODE_NODEPOOL}; do
  gcloud container node-pools create ${POOL} \
      --project=${PROJECT_ID} \
      --location=${REGION} \
      --node-locations=${ZONE} \
      --num-nodes=4 \
      --machine-type=tpu7x-standard-4t \
      --tpu-topology=2x2x4 \
      --cluster=${CLUSTER_NAME}
done
```

---

## Step 3: Build the Container Image

Clone `vllm-project/vllm-torchtpu` and build the `--target dev` image with `--vllm-rs` (which compiles the `vllm-rs` Rust frontend alongside `vllm-router` and the benchmarking tools):

```shell
rm -rf /tmp/vllm-torchtpu
git clone https://github.com/vllm-project/vllm-torchtpu.git /tmp/vllm-torchtpu
cd /tmp/vllm-torchtpu

gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
./docker/build_image.sh -t ${IMAGE_URI} --target dev --vllm-rs
docker push ${IMAGE_URI}
cd -
```

---

## Step 4: Pre-shard the Weights and Upload to GCS

The megakernel loads weights from a per-rank pre-sharded layout. Convert the checkpoints once on a CPU machine (no TPU needed):

```shell
export WORK_DIR="/mnt/disks/kimi_k3"
export DSPARK_REV="cf6b8244620e7ea4b0651d214f28e89eac75bed6"
mkdir -p ${WORK_DIR}

docker run --rm -v ${WORK_DIR}:/work -e DSPARK_REV=${DSPARK_REV} ${IMAGE_URI} bash -c '
  set -euo pipefail
  DSPARK=/work/models--Inferact--Kimi-K3-DSpark/snapshots/${DSPARK_REV}
  hf download moonshotai/Kimi-K3 --local-dir /work/Kimi-K3
  hf download Inferact/Kimi-K3-DSpark --revision ${DSPARK_REV} --local-dir ${DSPARK}
  cd /workspace/vllm-torchtpu
  export MEGAKERNEL_TOOL_WORKER=1
  python -m tools.megakernel preshard k3 --checkpoint /work/Kimi-K3 --output /work/presharded/kimi-k3-tp32 --ranks "$(seq -s, 0 31)"
  python -m tools.megakernel preshard prefill --checkpoint /work/Kimi-K3 --output /work/presharded/kimi-k3-tp32
  python -m tools.megakernel preshard dspark --draft ${DSPARK} --output /work/presharded/kimi-k3-tp32
  python -m tools.megakernel preshard verify --against /work/presharded/kimi-k3-tp32 --checkpoint /work/Kimi-K3 --draft ${DSPARK}
'
```

> [!IMPORTANT]
> Keep the DSpark snapshot directory named `cf6b8244620e7ea4b0651d214f28e89eac75bed6`. The loader derives the pre-shard subdirectory (`dspark-cf6b8244`) from it.

Upload the pre-shards (~1.55 TiB), the DSpark snapshot, and the Kimi-K3 config and tokenizer files to a bucket in the same region as the cluster:

```shell
gcloud storage buckets create gs://${GCS_MODEL_BUCKET} \
    --project=${PROJECT_ID} --location=${REGION} --uniform-bucket-level-access

gcloud storage cp ${WORK_DIR}/Kimi-K3/*.json ${WORK_DIR}/Kimi-K3/*.py ${WORK_DIR}/Kimi-K3/tiktoken.model \
    gs://${GCS_MODEL_BUCKET}/Kimi-K3/
gcloud storage cp -r ${WORK_DIR}/models--Inferact--Kimi-K3-DSpark gs://${GCS_MODEL_BUCKET}/
gcloud storage cp -r ${WORK_DIR}/presharded gs://${GCS_MODEL_BUCKET}/
```

The bucket must contain:

```text
gs://<YOUR_GCS_MODEL_BUCKET>/
├── Kimi-K3/                      # config.json, tiktoken.model, tokenization_kimi.py, ...
├── models--Inferact--Kimi-K3-DSpark/snapshots/cf6b8244620e7ea4b0651d214f28e89eac75bed6/
└── presharded/kimi-k3-tp32/      # layout.json, rank00..rank31/, prefill-sp-cache/, dspark-cf6b8244/
```

---

## Step 5: Setup Kubernetes Configurations

1. Configure `kubectl` and create the namespace:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID}
kubectl create namespace ${NAMESPACE}
```

2. Grant the namespace's `default` Kubernetes service account read access to the bucket (both roles are required):

```shell
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')
MEMBER="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/default"
for ROLE in roles/storage.objectViewer roles/storage.legacyBucketReader; do
  gcloud storage buckets add-iam-policy-binding gs://${GCS_MODEL_BUCKET} --member="${MEMBER}" --role="${ROLE}"
done
```

---

## Step 6: Apply the vLLM Server Manifest

From this recipe's directory (`inference/ironwood/vLLM/vLLM-TorchTPU/Kimi-K3`), apply [kimi_k3_megakernel-1p1d.yaml](./kimi_k3_megakernel-1p1d.yaml) with your image registry and bucket substituted:

```shell
sed -e "s|<REGISTRY>|${IMAGE_REGISTRY}|g" \
    -e "s|<YOUR_GCS_MODEL_BUCKET>|${GCS_MODEL_BUCKET}|g" \
    kimi_k3_megakernel-1p1d.yaml | kubectl apply -f -
```

The manifest creates two `LeaderWorkerSet`s (`vllm-kimi-k3-prefill` and `vllm-kimi-k3-decode`, 4 pods each, one per TPU host), a `vllm-router` `Deployment`, and a `ClusterIP` `Service` for each. The router starts serving once both engines are healthy.

Monitor the progress of the deployment:

```shell
kubectl get pods -n ${NAMESPACE} -w
```

Check the server logs:

```shell
kubectl logs -n ${NAMESPACE} vllm-kimi-k3-prefill-0 -c vllm-torchtpu -f
kubectl logs -n ${NAMESPACE} vllm-kimi-k3-decode-0 -c vllm-torchtpu -f
```

Startup takes about 10 minutes (weight load from GCS and kernel compilation). Once both engines are ready you should see:

```
(APIServer pid=...) INFO:     Application startup complete.
```

---

## Step 7: Interact with the Model (Optional)

1. Port forward the router service to port `8000` on your local machine:

```shell
kubectl port-forward -n ${NAMESPACE} service/vllm-kimi-k3-router-service 8000:8000
```

2. Run a `curl` request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "moonshotai/Kimi-K3",
        "messages": [
            {"role": "user", "content": "Explain in three bullet points how Mixture-of-Experts routing works."}
        ],
        "max_tokens": 160,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": false}
    }'
```

> [!NOTE]
> The megakernel sampler rejects `logprobs`, `prompt_logprobs`, `min_p`, `logit_bias`, `min_tokens`, and frequency, presence, and repetition penalties with `HTTP 400`.

---

## Step 8: Run Serving Benchmarks

We use `vllm bench serve` to measure an **8K Input / 1K Output (8k/1k)** workload at concurrency 1 and 16.

Save the following manifest as `kimi_k3_megakernel-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kimi-k3-mk-bench-8k1k
  namespace: vllm-kimi-k3-mk
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  restartPolicy: Never
  enableServiceLinks: false
  containers:
  - name: vllm-bench
    image: <REGISTRY>/vllm-tpu:k3-megakernel
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -euo pipefail
      # Run vLLM's CLI on the CPU platform: this pod has no TPU attached.
      export VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1
      until curl -sf http://vllm-kimi-k3-router-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      for C in 1 16; do
        N=$(( C == 1 ? 8 : 16 ))
        vllm bench serve \
          --backend=vllm \
          --base-url=http://vllm-kimi-k3-router-service:8000 \
          --model=moonshotai/Kimi-K3 \
          --tokenizer=/models/Kimi-K3 \
          --trust-remote-code \
          --dataset-name=random \
          --random-input-len=8192 \
          --random-output-len=1024 \
          --random-range-ratio=0 \
          --num-prompts=${N} \
          --max-concurrency=${C} \
          --temperature=0 \
          --ignore-eos \
          --percentile-metrics=ttft,tpot,itl,e2el
      done
    resources:
      requests:
        cpu: "8"
        memory: "24Gi"
    volumeMounts:
    - name: gcs-models
      mountPath: /models
      readOnly: true
  volumes:
  - name: gcs-models
    csi:
      driver: gcsfuse.csi.storage.gke.io
      readOnly: true
      volumeAttributes:
        bucketName: "<YOUR_GCS_MODEL_BUCKET>"
        mountOptions: "implicit-dirs"
```

Apply it and follow the logs:

```shell
sed -e "s|<REGISTRY>|${IMAGE_REGISTRY}|g" \
    -e "s|<YOUR_GCS_MODEL_BUCKET>|${GCS_MODEL_BUCKET}|g" \
    kimi_k3_megakernel-benchmark-8k1k.yaml | kubectl apply -f -
kubectl logs -n ${NAMESPACE} kimi-k3-mk-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete pod -n ${NAMESPACE} kimi-k3-mk-bench-8k1k
```

---

### Benchmark Results

Measured on `2 × 2x2x4` TPU v7x (32 chips) with this recipe, 8k/1k, `temperature=0`:

| Concurrency | Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Median TTFT (ms) | Median TPOT (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 8 | 346.83 | 3,121.50 | 455.92 | 2.24 |
| 16 | 16 | 1,225.33 | 11,027.97 | 1,851.05 | 8.04 |

Example output (concurrency 16):

```
============ Serving Benchmark Result ============
Successful requests:                     16
Failed requests:                         0
Maximum request concurrency:             16
Benchmark duration (s):                  13.37
Total input tokens:                      131072
Total generated tokens:                  16384
Request throughput (req/s):              1.20
Output token throughput (tok/s):         1225.33
Peak output token throughput (tok/s):    496.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          11027.97
---------------Time to First Token----------------
Mean TTFT (ms):                          1978.95
Median TTFT (ms):                        1851.05
P99 TTFT (ms):                           4255.38
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.59
Median TPOT (ms):                        8.04
P99 TPOT (ms):                           11.10
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.77
Median ITL (ms):                         31.50
P99 ITL (ms):                            34.78
----------------End-to-end Latency----------------
Mean E2EL (ms):                          10767.97
Median E2EL (ms):                        11271.29
P99 E2EL (ms):                           13228.23
==================================================
```

---

## Step 9: Cleanup

1. Delete the deployment:

```shell
kubectl delete namespace ${NAMESPACE}
```

2. (Optional) Delete the TPU node pools:

```shell
for POOL in ${PREFILL_NODEPOOL} ${DECODE_NODEPOOL}; do
  gcloud container node-pools delete ${POOL} \
      --cluster=${CLUSTER_NAME} \
      --location=${REGION} \
      --project=${PROJECT_ID}
done
```

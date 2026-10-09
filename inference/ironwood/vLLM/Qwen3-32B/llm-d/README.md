# Serve Qwen3-32B with llm-d on Ironwood TPU (optimized-baseline)

This recipe deploys [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) on
Ironwood (TPU7x) with vLLM behind the [llm-d](https://github.com/llm-d/llm-d)
router, following the upstream llm-d
[**optimized-baseline** guide](https://github.com/llm-d/llm-d/tree/77f18fe4f179ada1844d95ea1d4cb355a45b55b1/guides/optimized-baseline)
and its TPU v7 model-server overlay without modification.

The llm-d router (an EndpointPicker behind an Envoy proxy) places each request
on the vLLM replica that most likely holds its prompt prefix in cache, while
balancing token load across replicas.

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3-32B` |
| Accelerator | TPU7x (Ironwood), 2 single-host `2x2x1` slices (4 chips each) |
| Model server | vLLM (`docker.io/vllm/vllm-tpu:v0.29.0`), 2 replicas, `--tensor-parallel-size=8` |
| Router | llm-d router chart `v0.10.0`, optimized-baseline EndpointPicker config |
| llm-d guide revision | [`77f18fe`](https://github.com/llm-d/llm-d/tree/77f18fe4f179ada1844d95ea1d4cb355a45b55b1) |

All manifests and values files come from the llm-d repository at the pinned
revision; this recipe adds only the Ironwood cluster setup and the exact
variable values.

## Install client tools

You need `gcloud`, `kubectl`, `helm` (v3.12+), `git`, and `kustomize` support in
`kubectl` (built in). To install `gcloud`, follow
[Install the gcloud CLI](https://cloud.google.com/sdk/docs/install), then run
`gcloud auth login`.

## Cluster prerequisites

### Define parameters

```bash
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=<YOUR_REGION>
export ZONE=<YOUR_ZONE> # e.g., us-central1-c
export NODEPOOL_NAME=<YOUR_NODEPOOL_NAME>
export RESERVATION_NAME=<YOUR_RESERVATION_NAME> # Optional, if you have a reservation
```

### Create a cluster

Skip this step if you already have a GKE cluster.

```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --release-channel=rapid \
  --num-nodes=1
```

### Create the TPU7x node pool

The guide runs 2 vLLM replicas, each on one single-host TPU7x `2x2x1` slice
(`tpu7x-standard-4t`, 4 chips), so the node pool needs 2 nodes.

```bash
gcloud container node-pools create ${NODEPOOL_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --cluster=${CLUSTER_NAME} \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=2 \
  --reservation=${RESERVATION_NAME} \
  --reservation-affinity=specific
```

Configure `kubectl` and check that both nodes carry the labels the model
server selects on (`cloud.google.com/gke-tpu-accelerator=tpu7x`,
`cloud.google.com/gke-tpu-topology=2x2x1`):

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --location=${REGION} --project=${PROJECT_ID}

kubectl get nodes \
  -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology
```

## Deploy llm-d

### 1. Get the llm-d guide at the pinned revision

```bash
git clone https://github.com/llm-d/llm-d.git && cd llm-d
git checkout 77f18fe4f179ada1844d95ea1d4cb355a45b55b1
```

### 2. Set the guide variables

These are the upstream guide's variables with the TPU v7 values selected.

```bash
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
export GUIDE_NAME=optimized-baseline
export NAMESPACE=llm-d-optimized-baseline
export ACCELERATOR_TYPE=tpu/v7
export MODEL_SERVER=vllm
export MODEL=Qwen/Qwen3-32B
export MONITORING_VALUES=
export CURL_TEST_IMAGE=cfmanteiga/alpine-bash-curl-jq:latest
export HF_TOKEN=<YOUR_HF_TOKEN>

source ${REPO_ROOT}/guides/env.sh

# env.sh selects the floating `v0` router chart channel; pin the release this
# recipe was built against.
export ROUTER_CHART_VERSION=v0.10.0
```

`HF_TOKEN` must be a
[Hugging Face token](https://huggingface.co/docs/hub/security-tokens) with
read access to the model.

### 3. Install the Inference Extension CRDs, namespace, and token secret

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml

kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic llm-d-hf-token \
  --from-literal="HF_TOKEN=${HF_TOKEN}" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 4. Deploy the llm-d router (standalone mode)

```bash
helm install ${GUIDE_NAME} \
  ${ROUTER_STANDALONE_CHART} \
  -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
  ${MONITORING_VALUES} \
  -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

To front the router with a GKE Gateway instead, follow the guide's
[Gateway Mode](https://github.com/llm-d/llm-d/tree/77f18fe4f179ada1844d95ea1d4cb355a45b55b1/guides/optimized-baseline#1-deploy-the-llm-d-router)
section with `PROVIDER_NAME=gke`.

### 5. Deploy the vLLM model server

This applies the guide's TPU v7 overlay
([`modelserver/tpu/v7/vllm`](https://github.com/llm-d/llm-d/tree/77f18fe4f179ada1844d95ea1d4cb355a45b55b1/guides/optimized-baseline/modelserver/tpu/v7/vllm)):
2 replicas of `vllm serve Qwen/Qwen3-32B --tensor-parallel-size=8`, each
requesting `google.com/tpu: 4` on a `tpu7x` `2x2x1` node.

```bash
kubectl apply -n ${NAMESPACE} \
  -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/
```

Wait until both replicas are ready (the first start downloads the weights and
compiles the model, which can take several minutes):

```bash
kubectl get pods -n ${NAMESPACE} -w
```

## Verify

```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} \
  -o jsonpath='{.spec.clusterIP}')

kubectl run curl-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'curl -sS -X POST "http://${IP}/v1/completions" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"prompt\": \"How are you today?\"}"'
```

## Benchmark

The guide benchmarks with
[`llmdbenchmark`](https://github.com/llm-d/llm-d-benchmark) and
[`inference-perf`](https://github.com/kubernetes-sigs/inference-perf), using the
guide's dedicated workload profile:

```bash
export BENCHMARK_REF=main
export HARNESS=inference-perf
export WORKLOAD=guide_optimized-baseline_1.yaml
export GATEWAY_CLASS=epponly

curl -sSL https://raw.githubusercontent.com/llm-d/llm-d-benchmark/${BENCHMARK_REF}/install.sh | bash
cd llm-d-benchmark && source .venv/bin/activate

export ENDPOINT_URL="http://$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')"

llmdbenchmark \
  --spec guides/${GUIDE_NAME} \
  run \
  --endpoint-url "${ENDPOINT_URL}" \
  --gateway-class "${GATEWAY_CLASS}" \
  --model "${MODEL}" \
  --namespace "${NAMESPACE}" \
  --harness "${HARNESS}" \
  --workload "${WORKLOAD}" \
  --analyze
```

Results are written to the workspace directory printed by the CLI. See the
guide's [benchmarking notes](https://github.com/llm-d/llm-d/blob/77f18fe4f179ada1844d95ea1d4cb355a45b55b1/helpers/benchmark.md)
for other workload profiles and timeout settings.

> [!NOTE]
> The router's `prefix-cache-affinity-filter` uses its built-in
> `peakPrefillThroughput`, which upstream calibrated for H100. For best results
> on TPU7x, measure it with the guide's
> [calibration recipe](https://github.com/llm-d/llm-d/tree/77f18fe4f179ada1844d95ea1d4cb355a45b55b1/guides/recipes/router/calibration)
> and set it in `router/optimized-baseline.values.yaml`.

## Cleanup

```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}
kubectl delete -n ${NAMESPACE} \
  -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/
kubectl delete namespace ${NAMESPACE}
```

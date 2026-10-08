# Qwen3.5-397B-A17B-FP8 on Ironwood (tpu7x) with vLLM TorchTPU: multi-host 1P1D disaggregated serving, AgentX benchmark

This recipe serves `Qwen/Qwen3.5-397B-A17B-FP8` with the vLLM **TorchTPU**
backend using prefill/decode (P/D) disaggregation across two tpu7x hosts on GKE,
and benchmarks it with the InferenceX **AgentX** agentic-coding workload:

- **Prefill**: one tpu7x 2x2x1 host (4 chips), prefill context parallel 8
  (PCP8), prefix caching on.
- **Decode**: one tpu7x 2x2x1 host (4 chips), data parallel 8 (DP8).
- The TorchTPU Raiden KV connector (`TPURaidenConnector`) moves the KV cache
  from prefill to decode; a lightweight proxy routes each request to prefill and
  then decode.
- The client is the InferenceX AIPerf build running the
  `inferencex-agentx-mvp` scenario on the
  [`semianalysisai/cc-traces-weka-062126-256k`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)
  traces (multi-turn coding sessions, roughly 90k-104k input and ~900 output
  tokens per request), streaming, at concurrency 32 for 3600 s after a
  per-lane warmup.

## Prerequisites

Before running this recipe, update the placeholder values in `values.yaml`:

- `<YOUR_LOG_BUCKET>`: GCS bucket for job logs (e.g. `my-project-logs-bucket`).
- `<YOUR_RESERVATION>`: your GKE reservation name, if applicable.
- `<YOUR_MODEL_BUCKET>`: GCS bucket holding a copy of the
  `Qwen/Qwen3.5-397B-A17B-FP8` Hugging Face checkpoint under
  `Qwen3.5-397B-A17B-FP8/`. The engines stream the weights from GCS
  (`--load-format=runai_streamer`), so the pods' service account needs read
  access to this bucket.

Images (replace them with images your cluster can pull):

- Prefill, decode and proxy: a vLLM TorchTPU image (`prefill.image`,
  `decode.image`, `proxy.image`).
- Client: an InferenceX AgentX client image that provides the InferenceX AIPerf
  build under `/root/aiperf-venv` (`client.image`).

## Create the GKE Cluster
Create your tpu7x cluster using [XPK](https://github.com/AI-Hypercomputer/xpk).
The next sections assume you have created a cluster with tpu7x
nodes.


## Deploy vLLM Workload on GKE

### Configure kubectl to communicate with your cluster

```
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${LOCATION}
```

### Generate a new Hugging Face token if you don't already have one

On the huggingface website, create an account if necessary, and go to Your
Profile > Settings > Access Tokens.
Select Create new token.
Specify a name of your choice and a role with at least Read permissions.
Select Generate a token, follow the prompts and save your token.

(NOTE: Also ensure that your account has access to the model and the dataset on
Hugging Face.)

### Run the benchmark
In this directory, run:

```
helm install ${RUN_NAME} . --set hf_token=${HF_TOKEN}
```

The benchmark launches prefill and decode server pods, a proxy pod and a client
job. Every object in the recipe is labelled with
`app.kubernetes.io/instance=${RUN_NAME}`. To see what was created:

```
$ kubectl get pods -l app.kubernetes.io/instance=${RUN_NAME}
```

A cold start (weight streaming plus compilation) takes up to about an hour
before the engines report ready. On the server pods, at the end of the startup
you'll see logs such as:

```
$ kubectl logs -f --prefix --all-containers --max-log-requests=100 -l app.kubernetes.io/instance=${RUN_NAME},app.kubernetes.io/component=server

(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

The client waits until the proxy answers, warms up each lane, runs the 3600 s
measurement window, and then prints the AIPerf summary (request counts, total
and output token throughput, TTFT and inter-token latency percentiles, and the
server prefix-cache hit rate):

```
$ kubectl logs -f --all-containers -l app.kubernetes.io/instance=${RUN_NAME},app.kubernetes.io/component=client
```


### Customizing the benchmark

To change the benchmark parameters, such as the concurrency or the measurement
window, modify the client command in `values.yaml`.

### Cleanup
When you are done running the benchmark, the server will still be running. You
can cleanup by running:

```
helm uninstall ${RUN_NAME}
```

Helm does not delete volumes provisioned through `volumeClaimTemplates`, so
some PersistentVolumeClaims can outlive the release and keep consuming disk
quota. Reap any leftovers with:

```
kubectl delete pvc -l app.kubernetes.io/instance=${RUN_NAME}
```

# Qwen3.5-397B-A17B-FP8 on Ironwood (tpu7x) with vLLM TorchTPU: multi-host, AgentX benchmark

> [!IMPORTANT]
> This recipe is functional, not performance-optimized. It shows how to run
> one vLLM TorchTPU engine across the two hosts of a tpu7x slice. The server
> flags are chosen to start reliably, not tuned for throughput or latency.

This recipe serves `Qwen/Qwen3.5-397B-A17B-FP8` with the vLLM **TorchTPU**
backend on one tpu7x 2x2x2 slice on GKE, and benchmarks it with the
InferenceX **AgentX** agentic-coding workload.

## Topology

- One tpu7x 2x2x2 slice: 2 hosts, 4 chips per host, 8 chips and 16 TPU
  devices in total.
- One engine spans both hosts: tensor parallelism 16 (TP16) with expert
  parallelism, so the MoE experts are spread over the same 16 TPU devices
  (EP16). The server exposes one OpenAI-compatible endpoint.
- The two hosts run as one
  [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) group. The first
  pod starts a Ray head and `vllm serve`; the second pod joins the Ray
  cluster, which places the engine's TPU workers on both hosts.
- The server pods run privileged, as the upstream vLLM TorchTPU multi-host
  launch scripts do.

The client is the InferenceX AIPerf build running the `inferencex-agentx-mvp`
scenario on the
[`semianalysisai/cc-traces-weka-062126-256k`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)
traces (multi-turn coding sessions with long, mostly cached prompts),
streaming, at concurrency 32 for 3600 s after a per-lane warmup.

## Prerequisites

- A GKE cluster with a tpu7x node pool of topology 2x2x2 and the
  [LeaderWorkerSet](https://lws.sigs.k8s.io/docs/installation/) controller
  installed.
- A Hugging Face token with read access to `Qwen/Qwen3.5-397B-A17B-FP8` and
  to the AgentX dataset.
- Images your cluster can pull. The defaults in `values.yaml` are the images
  this recipe was validated with; replace them if your cluster cannot pull
  them:
  - `server.image`: a vLLM TorchTPU image. The default is the public
    `vllm/vllm-torchtpu:v0.29.0` release from Docker Hub; override it to use
    a newer release or a mirror in your own registry.
  - `client.image`: an InferenceX AgentX client image that provides the
    InferenceX AIPerf build under `/root/aiperf-venv`.

Optional settings in `values.yaml`, empty by default:

- `reservation`: the name of your tpu7x reservation.
- `kueue.queue_name` and `kueue.priority_class`: set these if your cluster
  admits workloads through Kueue.

## Create the GKE Cluster

Create your tpu7x cluster using [XPK](https://github.com/AI-Hypercomputer/xpk).
The next sections assume you have created a cluster with a tpu7x 2x2x2 node
pool.

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

The recipe creates one LeaderWorkerSet group of two server pods (one per
host), a Service in front of the first pod, and a client Job. Every object is
labelled with `app.kubernetes.io/instance=${RUN_NAME}`, and the pods are
additionally labelled `app.kubernetes.io/component=server` or
`app.kubernetes.io/component=client`. To see what was created:

```
kubectl get pods -l app.kubernetes.io/instance=${RUN_NAME}
```

Each server pod first downloads the checkpoint (about 400 GB) from Hugging
Face to its own data disk. Then the second pod joins the Ray cluster, and the
first pod starts `vllm serve`. A cold start (download plus compilation) can
take more than an hour. To follow the server logs:

```
kubectl logs -f --prefix --all-containers --max-log-requests=10 \
  -l app.kubernetes.io/instance=${RUN_NAME},app.kubernetes.io/component=server
```

When the server is ready, the first pod logs:

```
(APIServer pid=...) INFO:     Application startup complete.
```

The client waits until the server answers a chat completion, warms up each
lane, runs the 3600 s measurement window, and then prints the AIPerf summary
(request counts, token throughput, time to first token and inter-token
latency percentiles):

```
kubectl logs -f --all-containers \
  -l app.kubernetes.io/instance=${RUN_NAME},app.kubernetes.io/component=client
```

### Customizing the benchmark

All server and client settings live in `values.yaml`:

- `server.env` and `server.args`: TorchTPU settings and `vllm serve` flags.
  Keep `--tensor-parallel-size` (times `--data-parallel-size`, if you add
  it) equal to the 16 TPU devices of the slice, and keep `client.concurrency`
  at or below `--max-num-seqs`.
- `client.concurrency`, `client.duration_seconds`,
  `client.warmup_requests_per_lane`: the AgentX load. A window shorter than
  900 s is allowed but is not a canonical AgentX run.

### Cleanup

When you are done running the benchmark, the server will still be running.
Clean up everything the recipe created with:

```
helm uninstall ${RUN_NAME}
```

The data disks are deleted together with the LeaderWorkerSet. If any
PersistentVolumeClaim is left over, delete it with:

```
kubectl get pvc | grep ${RUN_NAME}-vllm-server
kubectl delete pvc data-volume-${RUN_NAME}-vllm-server-0 data-volume-${RUN_NAME}-vllm-server-0-1
```

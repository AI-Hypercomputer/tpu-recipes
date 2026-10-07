# Qwen3.5-397B-A17B-FP8 on Ironwood (tpu7x) with vLLM TorchTPU: single host

This recipe serves `Qwen/Qwen3.5-397B-A17B-FP8` with the vLLM **TorchTPU**
backend on one tpu7x 2x2x1 host (4 chips) on GKE, and benchmarks it with the
InferenceX serving client: random dataset, ISL/OSL 8192/1024, 640 prompts at
concurrency 64.

The server image is set in `values.yaml` (`server.image`, default
`us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:nightly-20261005`). Replace it with a TorchTPU vLLM image your cluster can pull.

### Server configuration

The server settings follow the InferenceX Qwen3.5-397B TorchTPU single-host
configuration for concurrency 64:

- `--tensor-parallel-size=8 --data-parallel-size=1 --enable-expert-parallel`
  (8 TPU cores across the 4 chips)
- `--max-num-seqs=64 --max-num-batched-tokens=16384 --max-model-len=10240`
- `--prefill-schedule-interval=1 --async-scheduling --block-size=256`
- FP8 weights and FP8 KV cache, prefix caching disabled
- TorchTPU env: `USE_MOE_FUSED_EP_KERNEL=0`, `USE_MOE_COUNTING_SORT=1`,
  `TPU_ENABLE_GDN_DYNAMIC_TILING=0`, `TPU_RPA_FOLD_KV_HEAD_DIM=1`,
  `TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS=1024`,
  `ONEHOT_MOE_PERMUTE_THRESHOLD=2048`

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

(NOTE: Also ensure that your account has access to the model on Hugging Face.
For example, for llama3, you will need to explicitly get permission):

### Run the benchmark
In this directory, run:

```
helm install ${RUN_NAME} . --set hf_token=${HF_TOKEN}
```

The benchmark launches one server pod and one client pod. Every object in the recipe
is labelled with `app.kubernetes.io/instance=${RUN_NAME}`, and the pods are
additionally labelled `app.kubernetes.io/component=server` or
`app.kubernetes.io/component=client`. To see what was created:

```
$ kubectl get pods -l app.kubernetes.io/instance=${RUN_NAME}
```

On the server pod, at the end of the server startup you'll see logs such as:

```
$ kubectl logs -f --prefix --all-containers --max-log-requests=100 -l app.kubernetes.io/instance=${RUN_NAME},app.kubernetes.io/component=server

(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

The client pod will wait until the server is up and then start the benchmark.
On the client pod, you'll see logs such as:

```
$ kubectl logs -f --prefix --all-containers -l app.kubernetes.io/instance=${RUN_NAME},app.kubernetes.io/component=client

============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Benchmark duration (s):                  xx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              xx
Output token throughput (tok/s):         xxx
Peak output token throughput (tok/s):    xxx
Peak concurrent requests:                10.00
Total Token throughput (tok/s):          xxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxx
Median TTFT (ms):                        xxx
P99 TTFT (ms):                           xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                       xxx
P99 TPOT (ms):                           xxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P99 ITL (ms):                            xxx
==================================================
```


### Customizing the benchmark

All server and client settings live in `values.yaml`:

- `server.image` / `client.image`: the TorchTPU vLLM image. The default is a
  dated nightly build this recipe was validated with; switch to an official
  vLLM TorchTPU release image once one is published.
- `server.env` and `server.args`: TorchTPU tuning and `vllm serve` flags.
- `client.args`: InferenceX `benchmark_serving.py` flags (dataset, ISL/OSL,
  number of prompts, concurrency).
- `reservation` (optional): your tpu7x GCE reservation name.
- `kueue.queue_name` / `kueue.priority_class` (optional): set these if your
  cluster admits workloads through Kueue; leave them empty otherwise.

Compiled artifacts are cached on the server's data disk
(`/data/torch_compile_cache`, `/data/xla_cache`), so no GCS bucket is needed.

### Cleanup

When you are done running the benchmark, the server will still be running.
Clean up everything the recipe created, including the data disk, with:

```
helm uninstall ${RUN_NAME}
```

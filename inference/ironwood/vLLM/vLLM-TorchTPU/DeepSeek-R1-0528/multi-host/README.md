
# Run vLLM (TorchTPU) torchtpu_vllm/ts_repro_deepseek_r1_0528_tp16_c64_v029 on tpu7x on GKE

This recipe covers running a vLLM inference workload on tpu7x on
GKE.

This recipe uses the **TorchTPU** vLLM backend. The server container is named `vllm-torchtpu` on single-host recipes and `vllm-server` on multi-host (LeaderWorkerSet) recipes; use the label selectors below to view its logs regardless of topology.

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
Note: Please search and replace `<YOUR_BUCKET_NAME>` in `values.yaml` with your actual Google Cloud Storage bucket name to store logs and cache.

In this directory, run:

```
helm install ${RUN_NAME} . --set hf_token=${HF_TOKEN}
```

The benchmark will launch a client and a server pod. Every object in the recipe
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

In order to change the parameters of the benchmark, such as the model, number of
prompts, etc, you can modify the settings in values.yaml.

### Cleanup
When you are done running the benchmark, the server will still be running. You
can cleanup by running:

```
helm uninstall ${RUN_NAME}
```

Helm does not delete volumes provisioned through `volumeClaimTemplates`, so on
multi-host recipes some PersistentVolumeClaims outlive the release and keep
consuming disk quota. Reap any leftovers with:

```
kubectl delete pvc -l app.kubernetes.io/instance=${RUN_NAME}
```

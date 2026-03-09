
# Run vLLM qwen3_coder_480b_a35b_instruct on tpu7x on GKE

This recipe covers running a vLLM inference workload on tpu7x on
GKE.

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

> [!IMPORTANT]
> Before running the Helm install, make sure you update the following placeholders in `values.yaml`:
> - `gcp_service_account`: Set this to your GCP service account that has access to the model GCS bucket.
> - `server.model`: Set this to the GCS path of your model (e.g., `gs://my-bucket/model`) or a Hugging Face model ID.

The benchmark will launch a client and server pod.

On the server pod, at the end of the server startup you’ll see logs such as:

```
$ kubectl logs deployment/vllm-tpu -f

(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

The client pod will wait until the server is up and then start the benchmark.
On the client pod, you'll see logs such as:

```
$ kubectl logs -f vllm-bench

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

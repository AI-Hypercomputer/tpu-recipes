# Serve Llama-3.1-8B (or any other model) with vLLM on TPU VMs.

In this guide, we show how to serve Llama-3.1-8B or [any supported model](https://docs.vllm.ai/en/latest/models/supported_models.html) with vLLM engine.

## Step 0: Install `gcloud cli`

You can reproduce this experiment from your dev environment (e.g. your laptop). You need to install `gcloud` locally to complete this tutorial.

To install `gcloud cli` please follow this guide: [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

Once it is installed, you can login to GCP from your terminal with this command: `gcloud auth login`.

## Step 1: Create a v6e TPU instance

We create a single VM with 4 chips - if you need larger instances, you can set a different value for `--topology` such as `1x1` or `4x2`. 

To learn more about topologies: [v6e VM Types](https://cloud.google.com/tpu/docs/v6e#vm-types).

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone 
export PROJECT=your-tpu-project

# this command creates a tpu vm with 4 Trillium (v6e) chips - adjust it to suit your needs
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v6e --topology 2x2 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

## Step 2: ssh to the instance

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 3: Use the latest vllm docker image for TPU
We use a pinned image but you can change it to `vllm/vllm-tpu:nightly` to get the latest TPU nightly image.

```bash
export DOCKER_URI=vllm/vllm-tpu:e92694b6fe264a85371317295bca6643508034ef
```

## Step 4: Run the docker container in the TPU instance

```bash
sudo docker run -t --rm --name $USER-vllm --privileged -v /dev/shm:/dev/shm --shm-size 10gb -p 8000:8000 --entrypoint /bin/bash -it ${DOCKER_URI}
```

## Step 5: Set up env variables
Export your hugging face token along with other environment variables inside the container.

```bash
export HF_HOME=/dev/shm
export HF_TOKEN=<your HF token>
export TMPDIR=/dev/shm/ray
export RAY_TMPDIR=/dev/shm/ray
```

## Step 6: Serve the model

Now we serve the vllm server. Make sure you keep this terminal open for the entire duration of this experiment.

```bash
vllm serve "meta-llama/Llama-3.1-8B" --download_dir /dev/shm --swap-space 16 --disable-log-requests --tensor_parallel_size=4 --max-model-len=4096
```

It takes a few minutes depending on the model size to prepare the server - once you see the below snippet in the logs, it means that the server is ready to serve requests or run benchmarks:

```bash
INFO:     Started server process [7]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 7: Prepare the test environment

Open a new terminal to test the server and run the benchmark (keep the previous terminal open).

First, we ssh into the TPU vm via the new terminal:

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 8: access the running container

```bash
sudo docker exec -it $USER-vllm bash
```

## Step 9: Test the server.

Let's submit a test request to the server. This helps us to see if the server is launched properly and we can see legitimate response from the model.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B",
        "prompt": "I love the mornings, because ",
        "max_tokens": 200,
        "temperature": 0
    }'
```

## Step 9: Download the data

To kick off the benchmark, we need to download the `ShareGPT` dataset.

```bash
cd /dev/shm
mkdir data
cd data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

pip install datasets pandas
```

## Step 10:  Run the benchmarking

Finally, we are ready to run the benchmark:

```bash
cd /workspace/vllm

export HF_TOKEN=<your HF token>

python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model "meta-llama/Llama-3.1-8B"  \
    --dataset-name sharegpt \
    --dataset-path /dev/shm/data/ShareGPT_V3_unfiltered_cleaned_split.json  \
    --num-prompts 1000 \
    --sharegpt-output-len 256
```

The snippet below is what you’d expect to see - the numbers vary based on the vllm version, the model size and the TPU instance type/size.

```bash
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  38.48     
Total input tokens:                      224330    
Total generated tokens:                  133123    
Request throughput (req/s):              25.99     
Output token throughput (tok/s):         3459.46   
Total Token throughput (tok/s):          9289.10   
---------------Time to First Token----------------
Mean TTFT (ms):                          12778.77  
Median TTFT (ms):                        13841.77  
P99 TTFT (ms):                           28706.87  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          119.79    
Median TPOT (ms):                        64.58     
P99 TPOT (ms):                           2167.53   
---------------Inter-token Latency----------------
Mean ITL (ms):                           62.95     
Median ITL (ms):                         43.00     
P99 ITL (ms):                            196.68    
==================================================
```


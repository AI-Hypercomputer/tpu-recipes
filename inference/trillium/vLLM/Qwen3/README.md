# Serve Qwen3 with vLLM on TPU VMs

In this guide, we show how to serve [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).

## Step 0: Install `gcloud cli`

You can reproduce this experiment from your dev environment (e.g. your laptop). You need to install `gcloud` locally to complete this tutorial.

To install `gcloud cli` please follow this guide: [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

Once it is installed, you can login to GCP from your terminal with this command: `gcloud auth login`.

## Step 1: Create a v6e TPU instance

We create a single VM. For Qwen3-4B, 1 chip is sufficient and for the 32B model, at least 4 chips are required. If you need a different number of chips, you can set a different value for `--topology` such as `1x1`, `2x4`, etc.

To learn more about topologies: [v6e VM Types](https://cloud.google.com/tpu/docs/v6e#vm-types).

> **Note:** Acquiring on-demand TPUs can be challenging due to high demand. We recommend using [Queued Resources](https://cloud.google.com/tpu/docs/queued-resources) to ensure you get the required capacity.

### Option 1: Create an on-demand TPU VM

This command attempts to create a TPU VM immediately.

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

# this command creates a tpu vm with 4 Trillium (v6e) chips - adjust it to suit your needs
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v6e --topology 2x2 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

### Option 2: Use Queued Resources (Recommended)

With Queued Resources, you submit a request for TPUs and it gets fulfilled when capacity is available.

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
export QR_ID=your-queued-resource-id # e.g. my-qr-request

# This command requests a v6e-4 (4 chips). Adjust accelerator-type for different sizes.
# For 1 chip (Qwen3-4B), use --accelerator-type v6e-1.
gcloud alpha compute tpus queued-resources create $QR_ID \
    --node-id $TPU_NAME \
    --project $PROJECT --zone $ZONE \
    --accelerator-type v6e-4 \
    --runtime-version v2-alpha-tpuv6e
```

You can check the status of your request with:

```bash
gcloud alpha compute tpus queued-resources list --project $PROJECT --zone $ZONE
```

Once the state is `ACTIVE`, your TPU VM is ready and you can proceed to the next steps.

## Step 2: ssh to the instance

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 3: Use the latest vllm docker image for TPU

```bash
export DOCKER_URI=vllm/vllm-tpu:nightly
```

## Step 4: Run the docker container in the TPU instance

```bash
sudo docker run -it --rm --name $USER-vllm --privileged --net=host \
    -v /dev/shm:/dev/shm \
    --shm-size 100gb \
    -p 8000:8000 \
    --entrypoint /bin/bash ${DOCKER_URI}
```

> **Note:** 100GB should be sufficient for the 32B model. For the 4B model allocate at least 10GB for the weights. See [this guide](https://cloud.google.com/tpu/docs/attach-durable-block-storage) for attaching durable block storage to TPUs.

## Step 5: Set up env variables

Export your hugging face token along with other environment variables inside the container.

```bash
export HF_HOME=/dev/shm
export HF_TOKEN=<your HF token>
```

## Step 6: Serve the model

Now we start the vllm server. Make sure you keep this terminal open for the entire duration of this experiment.

```bash
export MAX_MODEL_LEN=4096
export TP=4 # number of chips

vllm serve Qwen/Qwen3-32B \
    --seed 42 \
    --disable-log-requests \
    --gpu-memory-utilization 0.98 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 256 \
    --tensor-parallel-size $TP \
    --max-model-len $MAX_MODEL_LEN
```

For the 4B model, we recommend `--max-num-batched-tokens 1024 --max-num-seqs 128`.

It takes a few minutes depending on the model size to prepare the server - once you see the `Application startup complete.` message in the logs, it means that the server is ready to serve requests or run benchmarks:

```bash
(APIServer pid=7) INFO:     Waiting for application startup.
(APIServer pid=7) INFO:     Application startup complete.
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

## Step 8: Access the running container

To run the benchmark and install dependencies, you first need to enter the running container.

```bash
sudo docker exec -it $USER-vllm bash
```

The following steps for testing and benchmarking should be executed from within this container shell.

## Step 9: Test the server

Let's submit a test request to the server. This helps us to see if the server is launched properly and we can see legitimate response from the model.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B",
        "prompt": "I love the mornings, because ",
        "max_tokens": 200,
        "temperature": 0
    }'
```

## Step 10: Preparing the test image

You will need to install datasets as it's not available in the base vllm image.

```bash
pip install datasets
```

## Step 11: Run the benchmarking

Finally, we are ready to run the benchmark:

```bash
export MAX_INPUT_LEN=1800
export MAX_OUTPUT_LEN=128
export HF_TOKEN=<your HF token>

cd /workspace/vllm

vllm bench serve \
    --backend vllm \
    --model "Qwen/Qwen3-32B"  \
    --dataset-name random \
    --num-prompts 1000 \
    --random-input-len=$MAX_INPUT_LEN \
    --random-output-len=$MAX_OUTPUT_LEN \
    --seed 100
```

The snippet below is what you’d expect to see - the numbers vary based on the vllm version, the model size and the TPU instance type/size.

```bash
============ Serving Benchmark Result ============
Successful requests:                     xxxxxxx
Benchmark duration (s):                  xxxxxxx
Total input tokens:                      xxxxxxx
Total generated tokens:                  xxxxxxx
Request throughput (req/s):              xxxxxxx
Output token throughput (tok/s):         xxxxxxx
Total Token throughput (tok/s):          xxxxxxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxxxxxx
Median TTFT (ms):                        xxxxxxx
P99 TTFT (ms):                           xxxxxxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxxxxxx
Median TPOT (ms):                        xxxxxxx
P99 TPOT (ms):                           xxxxxxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxxxxxx
Median ITL (ms):                         xxxxxxx
P99 ITL (ms):                            xxxxxxx
==================================================
```

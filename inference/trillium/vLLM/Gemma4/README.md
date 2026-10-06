# Serve Gemma 4 IT with vLLM on Trillium TPU VMs

In this guide, we show how to serve Gemma 4 IT models (e.g., `google/gemma-4-31B-it`) with vLLM on Trillium (TPU v6e) using Google Compute Engine (GCE).

> Note: These setup instructions are **specifically for Gemma 4** on TPU and may not work for other models as it uses custom wheels and source builds for vllm and transformers.

## Verified Models

The following larger Gemma 4 models are verified for deployment on TPU.

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :---- | :---- | :---- | :---- |
| Gemma 4 31B IT | 31B | 4× | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 26B-A4B IT (MoE) | 26B (4B active) | 4× | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |

> [!NOTE]
> Gemma 4 E2B IT, Gemma 4 E4B IT are currently not verified for TPU deployment.

### Known Limitations with Current TPU Verification

The current vLLM wheels/configurations for Gemma 4 on TPU do not yet support the following preview features:
- **Guided Generations with Structured Outputs** (e.g. JSON schema enforcement).
- **Advanced Reasoning Parser** improvements.
- **Multimodal Audio Inference** (Transformers audio pipeline integration).

Users should rely on standard text and image inference as verified in this guide.

## Step 0: Install `gcloud cli`

You can reproduce this experiment from your dev environment (e.g. your laptop). You need to install `gcloud` locally to complete this tutorial.

To install `gcloud cli` please follow this guide: [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

Once it is installed, you can login to GCP from your terminal with this command: `gcloud auth login`.

## Step 1: Create a v6e TPU instance

We create a single VM with 8 Trillium chips (topology `2x4`) as Gemma 4 31B IT benefits from 8-way tensor parallelism.

To learn more about topologies: [v6e VM Types](https://cloud.google.com/tpu/docs/v6e#vm-types).

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone 
export PROJECT=your-tpu-project

# This command creates a tpu vm with 8 Trillium (v6e) chips
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v6e --topology 2x4 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

## Step 2: ssh to the instance

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 3: Use the vLLM docker image for TPU

The team is building a docker image for the deployment. Check the latest tags here: [vllm/vllm-tpu tags](https://hub.docker.com/r/vllm/vllm-tpu/tags).

```bash
export DOCKER_URI=vllm/vllm-tpu:gemma4

```

## Step 4: Run the docker container in the TPU instance

```bash
sudo docker run -t --rm --name $USER-vllm --privileged --net=host -v /dev/shm:/dev/shm --shm-size 10gb --entrypoint /bin/bash -it ${DOCKER_URI}
```

## Alternative: Fast Track with Docker Compose

If you prefer using Docker Compose to handle environment variables and container setup in one go, you can use the provided model-specific compose files:
- **Dense Model (31B):** `docker-compose-gemma4-31B.yml`
- **MoE Model (26B-A4B):** `docker-compose-gemma4-26B-A4B.yml`

1. Edit the appropriate file (e.g. `docker-compose-gemma4-31B.yml`) to set your `HF_TOKEN`.
2. Clone the recipes repository and navigate to the Gemma 4 folder:

```bash
git clone https://github.com/AI-Hypercomputer/tpu-recipes.git
cd tpu-recipes/inference/trillium/vLLM/Gemma4
docker compose -f docker-compose-gemma4-31B.yml up -d
```

This replaces Steps 4, 5, and 6 by automatically starting the server in the background. Skip to **Step 7** to test the server.

---

## Step 5: Set up env variables

Export your hugging face token along with other environment variables inside the container.

```bash
export HF_HOME=/dev/shm
export HF_TOKEN=<your HF token>
```

## Step 6: Serve the model

Now we serve the vllm server. Make sure you keep this terminal open for the entire duration of this experiment.


```bash
export MAX_MODEL_LEN=16384
export TP=8 # number of chips

vllm serve google/gemma-4-31B-it --max-model-len $MAX_MODEL_LEN --tensor-parallel-size $TP --disable_chunked_mm_input --enable-auto-tool-choice --tool-call-parser gemma4

```

It takes about 10 minutes depending on the model size to prepare the server - once you see the below snippet in the logs, it means that the server is ready to serve requests:

```bash
INFO:     Started server process [x]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 7: Prepare the test environment

Open a new terminal to test the server (keep the previous terminal open).

First, we ssh into the TPU vm via the new terminal:

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 8: Access the running container

Depending on how you deployed the vLLM container, use one of the following methods to access the container's shell:

### Option A: If you used the `docker run` command (standard deployment)

```bash
sudo docker exec -it $USER-vllm bash
```

### Option B: If you used Docker Compose

1. List your running containers to find your container's name, for example, `gemma4-vllm-server-1`:

    ```bash
    sudo docker ps
    ```

2. Access the container shell by replacing `<container_name>` with the name you found in step 1:

    ```bash
    sudo docker exec -it <container_name> bash
    ```

## Step 9: Test the server (Text + Image)

Let's submit a test request to the server with an image.

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-31B-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?auto=format&fit=crop&w=800&q=80"
                        }
                    },
                    {
                        "type": "text", 
                        "text": "Describe the image above"
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "top_p": 1.0
    }'
```

## Step 10: Run Benchmarking

You can benchmark the serving performance using the built-in `vllm bench serve` tools inside the running container.

Access the running container if you haven't already. See step 8 for the exact command based on your deployment method. For example, if you deployed the container using the `docker run` command in Step 4, access the shell by running:
   
```bash
sudo docker exec -it $USER-vllm bash
```

### Option A: Standard Text Benchmarking

Run the benchmark with random text inputs:

```bash
vllm bench serve \
    --backend vllm \
    --model "google/gemma-4-31B-it" \
    --dataset-name random \
    --num-prompts 100 \
    --random-input-len 1024 \
    --random-output-len 128
```

### Option B: Multimodal (Image) Benchmarking

Run the benchmark with synthetic multimodal (image) traffic:

```bash
vllm bench serve \
    --omni \
    --backend openai-chat-omni \
    --model "google/gemma-4-31B-it" \
    --dataset-name random-mm \
    --num-prompts 100 \
    --limit-mm-per-prompt '{"image": 1}'
```

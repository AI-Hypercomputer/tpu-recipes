# Serve Qwen3.8-27B on Trillium (v6e) TPU VM with vLLM

This guide serves [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense 27B, hybrid linear-attention/full-attention architecture, Apache-2.0, ungated) on a v6e-4 TPU VM and benchmarks it with `vllm bench serve`.

> **Note:** Qwen3.8-27B support requires a vLLM TPU nightly image. This guide pins `vllm/vllm-tpu:nightly-20260816-11250b4-1d2d83a`, which is verified end-to-end below; a stable tag is expected to supersede it.

## Install `gcloud cli`

Follow the [installation guide](https://cloud.google.com/sdk/docs/install), then authenticate with `gcloud auth login`.

## Create a v6e TPU instance

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT \
  --accelerator-type=v6e-4 \
  --version=v2-alpha-tpuv6e
```

Add `--spot` for spot capacity, or `--reserved` with a reservation. On-demand v6e capacity can be hard to acquire; [queued resources](https://cloud.google.com/tpu/docs/queued-resources) are the recommended way to guarantee it:

```bash
gcloud compute tpus queued-resources create ${TPU_NAME}-qr \
  --node-id=$TPU_NAME --zone=$ZONE --project=$PROJECT \
  --accelerator-type=v6e-4 --runtime-version=v2-alpha-tpuv6e

# check status until state is ACTIVE (node provisioned):
gcloud compute tpus queued-resources describe ${TPU_NAME}-qr \
  --zone=$ZONE --project=$PROJECT --format='value(state.state)'
```

## SSH to the instance

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT
```

## Pull the vLLM TPU docker image

```bash
export IMAGE=vllm/vllm-tpu:nightly-20260816-11250b4-1d2d83a
sudo docker pull $IMAGE
```

## Write the serve script

The `--limit-mm-per-prompt` value is JSON. Passing it through nested shells (`ssh ... bash -c "docker ... bash -c '...'"`) strips the quotes and vLLM fails at startup with `Value image:0 cannot be converted`. Writing the command to a file and mounting it into the container avoids the problem entirely:

```bash
cat > /tmp/serve-qwen38.sh <<'EOF'
#!/bin/bash
exec vllm serve Qwen/Qwen3.8-27B \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
EOF
chmod +x /tmp/serve-qwen38.sh
```

> **Do not add `--language-model-only`.** Qwen3.8-27B is a multimodal architecture; current TPU builds precompile the vision encoder whenever the model is multimodal, so skipping the vision weights crashes warmup with `StageMissingLayer(vision_tower)`. Loading the vision tower (~0.8 GB) is harmless for text-only serving — `--limit-mm-per-prompt '{"image":0,"video":0}'` keeps multimodal inputs disabled at the API level.
>
> Environment variables from other recipes (for example the Qwen3.5-397B settings) are not needed for this model: results are identical with and without them on this image.

## Run the docker container and serve the model

```bash
sudo docker run -d --name vllm-serve --privileged --net=host \
  -v /dev/shm:/dev/shm --shm-size 60gb \
  -v /tmp/serve-qwen38.sh:/serve-qwen38.sh \
  -e HF_HOME=/dev/shm \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --entrypoint /bin/bash $IMAGE /serve-qwen38.sh
```

The model is ungated, so no `HF_TOKEN` is required (add `-e HF_TOKEN=...` for authenticated downloads). Weights (~54 GB) download into `/dev/shm` — RAM, not disk. The container bind-mounts the host's `/dev/shm` (`-v /dev/shm:/dev/shm`), which defaults to half of the v6e-4 host's 720 GB of RAM, so capacity is ample and the `--shm-size` flag is not the governing limit; the default boot disk needs no resizing. Time to serving on a fresh VM is roughly 15 minutes: image pull, weight download, and compilation. Follow progress with `sudo docker logs -f vllm-serve`; the server is ready when the log shows:

```
(APIServer pid=1) INFO:     Application startup complete.
```

## Test the server

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "In one short sentence: what is a TPU?"}],
    "max_tokens": 200
  }'
```

The response carries the answer in `content` and the thinking trace in `reasoning` (split by `--reasoning-parser qwen3`).

## Run the benchmarking

Access the running container and run the benchmark:

```bash
sudo docker exec -it vllm-serve bash -c \
  "vllm bench serve --backend vllm --model Qwen/Qwen3.8-27B \
   --dataset-name random --random-input-len 1024 --random-output-len 1024 \
   --num-prompts 128 --max-concurrency 64 --ignore-eos --seed 42"
```

Expected output (measured 2026-08-16 on v6e-4 with the pinned image; independently reproduced twice within 0.1%):

```
============ Serving Benchmark Result ============
Successful requests:                     128
Failed requests:                         0
Maximum request concurrency:             64
Benchmark duration (s):                  70.06
Total input tokens:                      131072
Total generated tokens:                  131072
Request throughput (req/s):              1.83
Output token throughput (tok/s):         1870.83
Peak output token throughput (tok/s):    2048.00
Peak concurrent requests:                84.00
Total token throughput (tok/s):          3741.66
---------------Time to First Token----------------
Mean TTFT (ms):                          1099.15
Median TTFT (ms):                        419.17
P99 TTFT (ms):                           3366.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.97
Median TPOT (ms):                        33.43
P99 TPOT (ms):                           33.65
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.97
Median ITL (ms):                         31.37
P99 ITL (ms):                            104.43
==================================================
```

Measured across concurrencies with the same protocol:

| max-concurrency | num-prompts | Output tok/s | Mean TPOT | Median TTFT |
|---:|---:|---:|---:|---:|
| 64 | 128 | 1,875 | 32.9 ms | 419 ms |
| 128 | 256 | 2,494 | 49.0 ms | 694 ms |
| 256 | 512 | 2,464 | 85.0 ms | 7,650 ms |

Throughput saturates near 2,500 tok/s from concurrency 128 and is insensitive to `--kv-cache-dtype` (fp8 vs bf16 within 0.5%). On current builds the bound is the gated-delta-rule (linear-attention) decode path rather than memory bandwidth, so these numbers are expected to improve with future kernel releases.

## Clean up

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --project=$PROJECT --quiet
```

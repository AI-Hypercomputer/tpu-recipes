#!/bin/bash
TPU_NAME="gemma-v6e-vm"
ZONE="us-east5-b"

echo "==========================================="
echo "Downloading benchmark dataset..."
echo "==========================================="
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="sudo docker exec vllm-gemma4 wget -qO ShareGPT.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

echo "==========================================="
echo "Running TEXT benchmark with Speculative Decoding..."
echo "==========================================="
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model google/gemma-4-31B-it \
  --dataset-path ShareGPT.json \
  --dataset-name sharegpt \
  --num-prompts 100 \
  --request-rate 10
"

echo "==========================================="
echo "Running MULTIMODAL benchmark with Speculative Decoding..."
echo "==========================================="
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model google/gemma-4-31B-it \
  --dataset-name random-mm \
  --num-prompts 100 \
  --random-mm-limit-mm-per-prompt '{\"image\": 1, \"video\": 0}'
"

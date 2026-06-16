#!/bin/bash
TPU_NAME="gemma-v6e-vm"
ZONE="us-east5-b"
echo "-> Copying secure HF token to VM..."
gcloud compute tpus tpu-vm scp hf_token.txt ${TPU_NAME}:/tmp/hf_token.txt --zone=${ZONE}
HF_TOKEN=$(gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="cat /tmp/hf_token.txt" 2>/dev/null)

echo "-> Stopping existing vLLM containers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command='sudo docker rm -f vllm-gemma4 || true'

echo "-> Booting today's nightly container with explicit host HF cache and JAX compilation cache mounts..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
mkdir -p /home/jawadamin_google_com/.cache/vllm/xla_cache
sudo docker run -d --name vllm-gemma4 --privileged --network host --shm-size 16g \
  -v /dev/shm:/dev/shm \
  -v /home/jawadamin_google_com/.cache/huggingface:/root/.cache/huggingface \
  -v /home/jawadamin_google_com/.cache/vllm/xla_cache:/root/.cache/vllm/xla_cache \
  -e HF_TOKEN='${HF_TOKEN}' \
  -e HF_HOME='/root/.cache/huggingface' \
  -e USE_BATCHED_RPA_KERNEL=1 \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
  -e SKIP_JAX_PRECOMPILE=0 \
  -e ATTN_BUCKETIZED_NUM_REQS=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e VLLM_USE_JAX_PROFILER_SERVER=1 \
  -e USE_JAX_PROFILER_SERVER=1 \
  -e JAX_PROFILER_SERVER_PORT=9999 \
  vllm/vllm-tpu:nightly-20260611-1043491-248e33c \
  sleep infinity
"

echo "-> Cleaning up temporary token files..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="rm -f /tmp/hf_token.txt"
rm -f hf_token.txt

echo "-> Hotpatching transformers from HF git source..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 pip install git+https://github.com/huggingface/transformers.git
"

echo "-> Patching processing_gemma4.py to bypass dummy MM validation error..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 python3 -c \"
path = '/usr/local/lib/python3.12/site-packages/transformers/models/gemma4/processing_gemma4.py'
with open(path, 'r') as f:
    text = f.read()
bad_str = 'raise ValueError(\n                    f\\\"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed.\\\"\n                )'
if bad_str in text:
    text = text.replace(bad_str, 'pass')
    with open(path, 'w') as f:
        f.write(text)
    print('Processor successfully patched!')
else:
    print('Processor warning: target validation block not found.')
\"
"

echo "-> Copying patch_gemma4_mtp.py to container and running it..."
gcloud compute tpus tpu-vm scp patch_gemma4_mtp.py ${TPU_NAME}:/tmp/patch_gemma4_mtp.py --zone=${ZONE}
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="sudo docker cp /tmp/patch_gemma4_mtp.py vllm-gemma4:/tmp/patch_gemma4_mtp.py && sudo docker exec vllm-gemma4 python3 /tmp/patch_gemma4_mtp.py"

echo "-> Patching tpu_runner.py to fix multimodal inputs for speculative decoding..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 python3 -c \"
path = '/workspace/tpu_inference/tpu_inference/runner/tpu_runner.py'
with open(path, 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'input_ids, inputs_embeds = self._get_input_ids_embeds' in line:
        lines[i] = line.replace('input_ids, inputs_embeds', 'forward_input_ids, inputs_embeds')
    elif 'self.kv_caches,' in lines[i-1] and 'input_ids,' in line and 'attn_metadata,' in lines[i+1]:
        lines[i] = line.replace('input_ids,', 'forward_input_ids,')
with open(path, 'w') as f:
    f.writelines(lines)
print('tpu_runner.py successfully patched!')
\"
"

echo "-> Copying modified configs.py to container..."
gcloud compute tpus tpu-vm scp configs.py ${TPU_NAME}:/tmp/configs.py --zone=${ZONE}
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="sudo docker cp /tmp/configs.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/kernels/experimental/batched_rpa/configs.py"

echo "-> Copying modified model_loader.py and weight_utils.py to container..."
gcloud compute tpus tpu-vm scp model_loader.py ${TPU_NAME}:/tmp/model_loader.py --zone=${ZONE}
gcloud compute tpus tpu-vm scp weight_utils.py ${TPU_NAME}:/tmp/weight_utils.py --zone=${ZONE}
gcloud compute tpus tpu-vm scp compilation_manager.py ${TPU_NAME}:/tmp/compilation_manager.py --zone=${ZONE}
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker cp /tmp/model_loader.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/models/common/model_loader.py
sudo docker cp /tmp/weight_utils.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/models/jax/utils/weight_utils.py
sudo docker cp /tmp/compilation_manager.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/runner/compilation_manager.py
"

echo "-> Copying patch_qwix.py to container and running it..."
gcloud compute tpus tpu-vm scp patch_qwix.py ${TPU_NAME}:/tmp/patch_qwix.py --zone=${ZONE}
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="sudo docker cp /tmp/patch_qwix.py vllm-gemma4:/tmp/patch_qwix.py && sudo docker exec vllm-gemma4 python3 /tmp/patch_qwix.py"

echo "-> Starting API server in FP8..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec -d vllm-gemma4 bash -c \"python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --speculative-config '{\\\"model\\\": \\\"google/gemma-4-31B-it-assistant\\\", \\\"num_speculative_tokens\\\": 5}' \
  --additional_config '{\\\"quantization\\\": { \\\"qwix\\\": { \\\"rules\\\": [{ \\\"module_path\\\": \\\".*\\\", \\\"weight_qtype\\\": \\\"float8_e4m3fn\\\", \\\"act_qtype\\\": \\\"float8_e4m3fn\\\"}]}}}' \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8 \
  --block-size 32 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 > /vllm.log 2>&1\"
"

echo "-> Polling for readiness..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command='
while [[ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models)" != "200" ]]; do
    sleep 30
    echo -n "."
done'
echo "Server is ready!"

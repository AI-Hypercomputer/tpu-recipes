# Setup

## Step 1: Download JetStream and MaxText github repository
```bash
cd ~
git clone https://github.com/google/maxtext.git
cd maxtext
git checkout main

cd ~
git clone https://github.com/google/JetStream.git
cd JetStream
git checkout main
```

## Step 2: Setup JetStream and MaxText
```bash
cd ~
sudo apt-get update
sudo apt install python3.10-venv
python -m venv venv-maxtext
source venv-maxtext/bin/activate

cd ~
cd JetStream
pip install -e .
cd benchmarks
pip install -r requirements.in

cd ~
cd maxtext/
bash setup.sh
```

## Step 3: Checkpoint conversion

```bash
# Go to https://llama.meta.com/llama-downloads/ and fill out the form
git clone https://github.com/meta-llama/llama
cd ~/maxtext/llama
bash download.sh # When prompted, choose 7B. This should create a directory llama-2-7b inside the llama directory


export CHKPT_BUCKET=gs://...
export MAXTEXT_BUCKET_SCANNED=gs://...
export MAXTEXT_BUCKET_UNSCANNED=gs://...
cd ~/maxtext
gsutil cp -r llama/llama-2-7b/* ${CHKPT_BUCKET}


# Checkpoint conversion
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh llama2 7b ${CHKPT_BUCKET} ${MAXTEXT_BUCKET_SCANNED} ${MAXTEXT_BUCKET_UNSCANNED}

# The path to the unscanned checkpoint should be set by the script, but set it explicitly if it hasn't
# For example export UNSCANNED_CKPT_PATH=gs://${MAXTEXT_BUCKET_UNSCANNED}/llama2-7b_unscanned_chkpt_2024-08-23-23-17/checkpoints/0/items
export UNSCANNED_CKPT_PATH=gs://..
```

# Benchmark

In terminal tab 1, start the server:
```bash
export TOKENIZER_PATH=assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=llama2-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11

cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
```

In terminal tab 2, run the benchmark:
```bash
source venv-maxtext/bin/activate

python JetStream/benchmarks/benchmark_serving.py   \
--tokenizer ~/maxtext/assets/tokenizer.llama2  \
--warmup-mode sampled   \
--save-result   \
--save-request-outputs   \
--request-outputs-file-path outputs.json   \
--num-prompts 1000   \
--max-output-length 1024   \
--dataset openorca
```

After the benchmark finishes, you should see something like 
```bash
Successful requests: 995
Benchmark duration: 305.366344 s
Total input tokens: 217011
Total generated tokens: 934964
Request throughput: 3.26 requests/s
Input token throughput: 710.66 tokens/s
Output token throughput: 3061.78 tokens/s
Mean TTFT: 130288.20 ms
Median TTFT: 140039.96 ms
P99 TTFT: 278498.91 ms
Mean TPOT: 5052.76 ms
Median TPOT: 164.01 ms
P99 TPOT: 112171.56 ms

```

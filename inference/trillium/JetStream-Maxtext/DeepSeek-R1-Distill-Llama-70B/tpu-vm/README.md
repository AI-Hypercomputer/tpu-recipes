# Inference benchmark of DeepSeek-R1-Distill-Llama-70B with JetStream MaxText Engine on a v6e (Trillium) TPU VM

This recipe outlines the steps to benchmark the inference of [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) model using [JetStream](https://github.com/AI-Hypercomputer/JetStream/tree/main) with [MaxText](https://github.com/AI-Hypercomputer/maxtext) engine on a [v6e TPU VM](https://cloud.google.com/tpu/docs/v6e-intro).

## Outline

1. [Provision a TPU v6e VM](#step-1-create-a-tpu-v6e-vm)
2. [Download JetStream and MaxText GitHub repository](#step-2-download-jetstream-and-maxtext-github-repository)
3. [Setup JetStream and MaxText](#step-3-setup-jetstream-and-maxtext)
4. [Configure environment variables](#step-4-configure-environment-variables)
5. [Convert Hugging Face Checkpoint to MaxText compatible checkpoint](#step-5-convert-hugging-face-checkpoint-to-maxtext-compatible-checkpoint)
6. [Unscan the checkpoint for efficient serving](#step-6-unscan-the-checkpoint-to-be-used-for-serving)
7. [Run MMLU benchmark](#step-7-run-mmlu-benchmark)
8. [Run MATH500 benchmark](#step-8-run-math500-benchmark)

## Step 1: Create a TPU v6e VM

This recipe sets up a v6e TPU VM as [queued resources](https://cloud.google.com/tpu/docs/queued-resources). Follow the steps in the documentation to [provision the Cloud TPU environment](https://cloud.google.com/tpu/docs/v6e-intro#provision-cloud-tpu).

Before provisioning the environment, verify all the [prerequisites](https://cloud.google.com/tpu/docs/v6e-intro/#provision-cloud-tpu) are met.

* Verify that your project has enough `TPUS_PER_TPU_FAMILY` quota, which specifies the maximum number of chips you can access within your Google Cloud project.
* Verify that your project has enough TPU quota for:
  * TPU VM quota
  * IP Address quota
  * Hyperdisk-balance quota

### Environment variables

In a Cloud Shell, create the following environment variables:

```
export PROJECT_ID="[your-project-id]"
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
export NODE_ID=inference-tpu-v6e-8
export ZONE=[supported-zone]
export ACCELERATOR_TYPE=v6e-8
export RUNTIME_VERSION=v2-alpha-tpuv6e
export SERVICE_ACCOUNT=${PROJECT_NUMBER}-compute@developer.gserviceaccount.com
export QUEUED_RESOURCE_ID=${NODE_ID}-${ZONE}
```

<details>
 <summary>Environment variables descriptions</summary>
 <li>`PROJECT_ID`: Google Cloud Project Name. Use an existing project or create a new one at</li>
 <li>`NODE_ID`: The user-assigned ID of the TPU which is created when the queued resource request is allocated.</li>
 <li>`ZONE`: See the [TPU regions and zones](https://cloud.google.com/tpu/docs/regions-zones) document for the supported zones.</li>
 <li>`ACCELERATOR_TYPE`: v6e-8</li>
 <li>`RUNTIME_VERSION`: v2-alpha-tpuv6e</li>
 <li>`SERVICE_ACCOUNT`: This is the email address for your service account that you can find in Google Cloud Console \-\> IAM \-\> Service Accounts. For example: tpu-service-account@\<your\_project\_ID\>.iam.gserviceaccount.com.com</li>
 <li>`QUEUED_RESOURCE_ID`: The user-assigned text ID of the queued resource request.</li>
</details>

* Create a Hyperdisk Balanced disk of size 500GB

By default, a TPU VM includes a 100GB boot disk. For working with DeepSeek-R1-Distill-Llama-70B, you would need additional storage for checkpoint conversion and processing. Create and attach a [Hyperdisk Balanced disk](https://cloud.google.com/compute/docs/disks/add-hyperdisk) of size 500GB to the TPU VM. Ensure you format and mount the disk.

```
gcloud compute disks create ${NODE_ID}-hd \
    --size 500GB  \
    --zone ${ZONE} \
    --type hyperdisk-balanced \
    --project ${PROJECT_ID}
```

* (optional) Use a custom network for better performance as well as to avoid having the default network becoming overloaded. Please refer to the [network performance optimizations](https://cloud.google.com/tpu/docs/v6e-intro/#network_performance_optimizations) for more details.

```
export NETWORK_NAME=${PROJECT_ID}-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-fw

gcloud compute networks create ${NETWORK_NAME} \
  --mtu=8896 \
  --project=${PROJECT_ID} \
  --subnet-mode=auto \
  --bgp-routing-mode=regional

gcloud compute firewall-rules create ${NETWORK_FW_NAME} \
  --network ${NETWORK_NAME} \
  --allow tcp,icmp,udp \
  --project=${PROJECT_ID}
```

* Create a Cloud Storage bucket to store model checkpoints. It's recommended to create a bucket in the same region as the TPU VM.

```
export GCS_REGION=eu
export GCS_BUCKET="[your-bucket-name]"

gcloud storage buckets create gs://${GCS_BUCKET} \
  --project=$PROJECT_ID \
  --location=$GCS_REGION \
  --uniform-bucket-level-access
```

* [Provision a v6e TPU VM](https://cloud.google.com/tpu/docs/v6e-intro/#provision-queued-resource) with 8 chips (v6e-8) attached with disk using queued resources.

```
gcloud alpha compute tpus queued-resources create $QUEUED_RESOURCE_ID \
  --node-id $NODE_ID \
  --project $PROJECT_ID \
  --zone $ZONE \
  --accelerator-type ${ACCELERATOR_TYPE} \
  --runtime-version ${RUNTIME_VERSION} \
  --service-account $SERVICE_ACCOUNT \
  --reserved \
  --network ${NETWORK_NAME} \
  --data-disk source=projects/$PROJECT_ID/zones/${ZONE}/disks/${NODE_ID}-hd,mode=read-write
```

* Connect to your TPU VMs using SSH

```
gcloud alpha compute tpus tpu-vm ssh ${NODE_ID} --project ${PROJECT_ID} --zone=${ZONE}
```

## Step 2: Download JetStream and MaxText GitHub repository

```
cd ~
git clone https://github.com/AI-Hypercomputer/maxtext.git
git clone https://github.com/AI-Hypercomputer/JetStream.git
```

## Step 3: Setup JetStream and MaxText

* Create a Python virtual environment

```
cd ~
sudo apt install python3.10-venv
python -m venv venv-maxtext
source venv-maxtext/bin/activate
```

* Install JetStream benchmarks dependencies

```
cd ~
cd JetStream
pip install -e .
cd benchmarks
pip install -r requirements.in
```

* Install maxtext

```
cd ~
cd maxtext/
bash setup.sh JAX_VERSION=0.5.0
```

* Install additional dependencies

```
pip install -U "huggingface_hub[cli]" hf_transfer scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m nltk.downloader punkt_tab

# Login to HuggingFace CLI
huggingface-cli login --token $HF_TOKEN
```

## Step 4: Configure environment variables

```
export LOCAL_DIR=/mnt/disks/persist
export HOME_DIR=$(bash -c "cd ~ && pwd")
export GCS_BUCKET="[your-bucket-name]"
export HF_HUB_ENABLE_HF_TRANSFER=1
export RUN_DATE=$(date +"%Y-%m-%d-%H-%M")
export RUN_NAME=unscanned_chkpt
export TOKENIZER_PATH=${HOME_DIR}/maxtext/assets/tokenizer_llama3.tiktoken
export CHECKPOINT_ORIGINAL=${LOCAL_DIR}/llama3.1-70b
export BASE_OUTPUT_PATH=gs://${GCS_BUCKET}/DeepSeek-R1-Distilled/maxtext/llama3.1/70b
export CHECKPOINT_TPU_SCANNED=${BASE_OUTPUT_PATH}/scanned_chkpt/${RUN_DATE}
export CHECKPOINT_TPU_UNSCANNED=${BASE_OUTPUT_PATH}/$RUN_NAME/checkpoints/0/items
export MODEL_NAME=llama3.1-70b

# Set up directories
sudo mkdir -p ${LOCAL_DIR}
```

## Step 5: Convert Hugging Face Checkpoint to MaxText compatible checkpoint

* Download checkpoint from hugging face

```
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B --local-dir $CHECKPOINT_ORIGINAL
```

* Convert Hugging Face to MaxText compatible checkpoint

```
cd ~
cd maxtext/
JAX_PLATFORMS=cpu python MaxText/llama_or_mistral_ckpt.py \
  --base-model-path $CHECKPOINT_ORIGINAL \
  --maxtext-model-path $CHECKPOINT_TPU_SCANNED \
  --model-size ${MODEL_NAME} \
  --huggingface-checkpoint true
```

After the checkpoint is converted to MaxText compatible format, you will see logs similar to the following

```
Loading the base model from /mnt/disks/persist/llama3.1-70b
Loading checkpoint 1 of 17 ...
Loading checkpoint 2 of 17 ...
Loading checkpoint 3 of 17 ...
Loading checkpoint 4 of 17 ...
Loading checkpoint 5 of 17 ...
Loading checkpoint 6 of 17 ...
Loading checkpoint 7 of 17 ...
Loading checkpoint 8 of 17 ...
Loading checkpoint 9 of 17 ...
Loading checkpoint 10 of 17 ...
Loading checkpoint 11 of 17 ...
Loading checkpoint 12 of 17 ...
Loading checkpoint 13 of 17 ...
Loading checkpoint 14 of 17 ...
Loading checkpoint 15 of 17 ...
Loading checkpoint 16 of 17 ...
Loading checkpoint 17 of 17 ...
Processing decoder norm scale
Processing logits dense
Processing token embeddings
Processing self attention
Processing pre and post self attention norms
Processing layer weights
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
sharding first axis
Creating checkpoint manager...
Checkpoint manager created!
I0222 00:27:10.324101 1251089 google_auth_provider.cc:181] Running on GCE, using service account xxxxxxxx@developer.gserviceaccount.com
saved a checkpoint at step 0
```

## Step 6: Unscan the checkpoint to be used for serving

```
cd ~
cd maxtext/
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py \
  MaxText/configs/base.yml \
  async_checkpointing=false \
  base_output_directory=${BASE_OUTPUT_PATH} \
  load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items \
  run_name=${RUN_NAME} \
  model_name=${MODEL_NAME} \
  force_unroll=true
```

After successful conversion, you will see logs similar to the following:

```
Running Model: llama3.1-70b
Updating following parameters in config

base_emb_dim: 8192
base_num_query_heads: 64
base_num_kv_heads: 8
base_num_decoder_layers: 80
base_mlp_dim: 28672
head_dim: 128
mlp_activations: ['silu', 'linear']
vocab_size: 128256
enable_dropout: False
logits_via_embedding: False
normalization_layer_epsilon: 1e-05
rope_max_timescale: 500000

...

Creating checkpoint manager...
...
Checkpoint manager created!
Read training checkpoint from:
checkpoint manager exists so trying to load this run's existing checkpoint
restoring params from gs://[your-bucket-name]/DeepSeek-R1-Distilled/maxtext/llama3.1/70b/scanned_chkpt/2025-02-20-03-44/0/items
....
I0222 01:13:43.526280 137903942428672 checkpoint_manager.py:1274] [process=0] Saving checkpoint at step 0
I0222 01:13:43.526471 137903942428672 checkpointer.py:216] [process=0] Started saving checkpoint to gs://[your-bucket-name]/DeepSeek-R1-Distilled/maxtext/llama3.1/70b/unscanned_chkpt/checkpoints/0.
...
I0222 01:16:28.086716 137903942428672 standard_logger.py:34] {'step': 0, 'event_type': 'save', 'directory': 'gs://[your-bucket-name]/DeepSeek-R1-Distilled/maxtext/llama3.1/70b/unscanned_chkpt/checkpoints', 'reached_preemption': False, 'preemption_received_at': None, 'synchronous': True, 'wait_for_prev_start_time': 1740186823.222154, 'wait_for_prev_duration_secs': 0.00030732154846191406, 'checkpointer_blocking_start_time': 1740186823.5264096, 'checkpointer_blocking_duration_secs': 164.55999898910522, 'get_old_steps_start_time': 1740186988.0864553, 'get_old_steps_duration_secs': 9.5367431640625e-06, 'checkpoint_manager_blocking_start_time': 1740186823.2219245, 'checkpoint_manager_blocking_duration_secs': 164.86473536491394}
saved an decode checkpoint at gs://[your-bucket-name]/DeepSeek-R1-Distilled/maxtext/llama3.1/70b/unscanned_chkpt/checkpoints/
I0222 01:16:28.086908 137903942428672 checkpoint_manager.py:1806] [process=0][thread=MainThread][wait_until_finished] No Save Finalize thread to wait for. Returning.
Successfully generated decode checkpoint at: gs://[your-bucket-name]/DeepSeek-R1-Distilled/maxtext/llama3.1/70b/unscanned_chkpt/checkpoints/0/items
```

* Verify the checkpoint by running the following command

```
cd ~
cd maxtext/
JAX_PLATFORMS=tpu python MaxText/decode.py \
  MaxText/configs/base.yml \
  tokenizer_path=$TOKENIZER_PATH \
  load_parameters_path=$CHECKPOINT_TPU_UNSCANNED \
  model_name=$MODEL_NAME \
  weight_dtype=bfloat16 \
  async_checkpointing=false \
  scan_layers=false \
  ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=1 \
  ici_tensor_parallelism=-1 \
  per_device_batch_size=1 \
  max_prefill_predict_length=64 \
  max_target_length=128 \
  attention=dot_product \
  optimize_mesh_for_tpu_v6e=True \
  run_name=runner_$(date +"%Y-%m-%d-%H-%M") \
  prompt="I love to"
```

You can see output similar to the following:

```
...

Memstats: After load_params:
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_0(process=0,(0,0,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_1(process=0,(1,0,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_2(process=0,(0,1,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_3(process=0,(1,1,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_4(process=0,(0,2,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_5(process=0,(1,2,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_6(process=0,(0,3,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_7(process=0,(1,3,0,0))
I0222 01:24:48.101375 140370795436032 llama3_tokenizer.py:87] Reloaded tiktoken model from assets/tokenizer_llama3.tiktoken
I0222 01:24:48.101573 140370795436032 llama3_tokenizer.py:99] #words: 128256 - BOS ID: 128000 - EOS ID: 128001
Input `I love to` -> ` read, but I don't have much time. How can I read more books?
I'm a busy person, but I want to read more. How can I fit reading into my schedule?
I want to read more, but I don't have enough time. What can I do?
I don't have time to read`
```

* Verify the checkpoint by running the following command with KV cache quantization

```
cd ~
cd maxtext/
JAX_PLATFORMS=tpu python MaxText/decode.py \
  MaxText/configs/base.yml \
  tokenizer_path=$TOKENIZER_PATH \
  load_parameters_path=$CHECKPOINT_TPU_UNSCANNED \
  model_name=$MODEL_NAME \
  weight_dtype=bfloat16 \
  async_checkpointing=false \
  scan_layers=false \
  ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=1 \
  ici_tensor_parallelism=-1 \
  per_device_batch_size=1 \
  max_prefill_predict_length=64 \
  max_target_length=128 \
  attention=dot_product \
  quantize_kvcache=true \
  optimize_mesh_for_tpu_v6e=True \
  run_name=runner_$(date +"%Y-%m-%d-%H-%M") \
  prompt="I love to"
```

You can see output similar to the following:

```
...

Memstats: After load_params:
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_0(process=0,(0,0,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_1(process=0,(1,0,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_2(process=0,(0,1,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_3(process=0,(1,1,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_4(process=0,(0,2,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_5(process=0,(1,2,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_6(process=0,(0,3,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_7(process=0,(1,3,0,0))
I0222 01:29:14.463668 124370055936000 llama3_tokenizer.py:87] Reloaded tiktoken model from assets/tokenizer_llama3.tiktoken
I0222 01:29:14.463912 124370055936000 llama3_tokenizer.py:99] #words: 128256 - BOS ID: 128000 - EOS ID: 128001
Input `I love to` -> ` read, but I don't have much time. How can I read more books?
I'm a busy person, but I want to read more. How can I fit reading into my schedule?
I want to read more, but I don't have enough time. What can I do?
I don't have time to read`
```

## Step 7: Run MMLU benchmark

* Download MMLU dataset

```
mkdir -p ${LOCAL_DIR}/mmlu
cd ${LOCAL_DIR}/mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P ${LOCAL_DIR}/mmlu
tar -xvf data.tar
```

### Start MaxText Engine server

To run the benchmark, open two SSH sessions connected to the TPU VM. In the first SSH session, start the MaxText Engine server with the following configuration:

```
cd ~
cd maxtext/
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=1536
export MODEL_NAME=llama3.1-70b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=2
export QUANTIZE_KVCACHE=True

# Set XLA flags
export LIBTPU_INIT_ARGS=""" --=false
--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false
--xla_tpu_enable_windowed_einsum_for_all_gather=false
--xla_tpu_prefer_latch_optimized_rhs_layouts=true
--xla_tpu_enable_experimental_fusion_cost_model=false
--xla_tpu_dot_dot_fusion_duplicated=false
--xla_tpu_dot_dot_fusion=true
--xla_jf_conv_input_fusion=true
--xla_jf_conv_output_fusion=true
--xla_tpu_rwb_fusion=false
--xla_tpu_copy_fusion_pad_unpad_ratio=0
--xla_tpu_licm_size_inflation_ratio=1
--xla_tpu_copy_elision_analysis_allowance=150000
--xla_tpu_copy_insertion_use_region_analysis_limit=10000
--xla_tpu_order_dot_after_layout=true
--xla_jf_rematerialization_percent_shared_memory_limit=100
--xla_tpu_use_repeated_instance_for_preferred_prefetch_time=true
--xla_tpu_enforce_prefetch_fifo_order=false
--xla_tpu_prefetch_interval_picker_size_override=6000000
--xla_tpu_async_copy_bandwidth_scaling_factor=1
--xla_tpu_nd_short_transfer_max_chunks=-1
--xla_tpu_enable_aggressive_broadcast_priority_update=true
--xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=SQRT
--xla_tpu_memory_bound_loop_optimizer_options=enabled:true
--xla_tpu_enable_copy_fusion=true
--xla_tpu_enable_cross_program_prefetch_freeing=false
--xla_tpu_enable_dot_strength_reduction=true
--xla_tpu_layout_use_dot_grouping=false
--xla_tpu_msa_inefficient_use_to_copy_ratio=0.5
--xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false
--xla_tpu_vector_load_fusion_window=1024
--xla_tpu_vector_store_fusion_window=256
--xla_jf_conv_reshape_fusion=false
--xla_tpu_input_conv_multi_users=false
--xla_tpu_enable_multi_level_input_dot_dot_fusion=false
--xla_tpu_enable_multi_level_output_dot_dot_fusion=false
--xla_tpu_dot_dot_fusion_separable_convs_only=false
--xla_tpu_enable_multi_level_nested_loop_fusion=true
--xla_tpu_nested_dot_fusion=true
--xla_tpu_enable_multi_level_nested_dot_fusion=false
--xla_jf_enable_multi_output_fusion=true
--xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions=false
--xla_tpu_enable_flash_attention=true
"""

python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${CHECKPOINT_TPU_UNSCANNED} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  attention=dot_product \
  optimize_mesh_for_tpu_v6e=True \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  quantize_kvcache=${QUANTIZE_KVCACHE}
```

After MaxText Engine server runs successfully, you will see logs similar to the following:

```
Running Model: llama3.1-70b
Updating following parameters in config

base_emb_dim: 8192
base_num_query_heads: 64
base_num_kv_heads: 8
...
Loading decode params from gs://[your-bucket-name]/DeepSeek-R1-Distilled/maxtext/llama3.1/70b/unscanned_chkpt/checkpoints/0/items
restoring params from gs://[your-bucket-name]/DeepSeek-R1-Distilled/maxtext/llama3.1/70b/unscanned_chkpt/checkpoints/0/items
I0224 04:24:46.351299 2083216 google_auth_provider.cc:181] Running on GCE, using service account xxxxxx@developer.gserviceaccount.com

Memstats: After load_params:
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_0(process=0,(0,0,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_1(process=0,(1,0,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_2(process=0,(0,1,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_3(process=0,(1,1,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_4(process=0,(0,2,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_5(process=0,(1,2,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_6(process=0,(0,3,0,0))
        Using (GB) 16.43 / 31.25 (52.576000%) on TPU_7(process=0,(1,3,0,0))
GC tweaked (allocs, gen1, gen2):  60000 20 30
```

### Run MMLU benchmarking

After MaxText Engine server is running, in the second SSH session to the TPU VM, start the MMLU benchmarking with following configuration:

```
cd ~
source venv-maxtext/bin/activate
cd ~/JetStream
JAX_PLATFORMS=tpu python benchmarks/benchmark_serving.py \
  --tokenizer=$TOKENIZER_PATH \
  --num-prompts 1000 \
  --dataset mmlu \
  --dataset-path ${LOCAL_DIR}/mmlu/data/test \
  --request-rate 0 \
  --warmup-mode sampled \
  --save-request-outputs \
  --num-shots=5 \
  --run-mmlu-dataset \
  --run-eval True \
  --model=llama-3 \
  --save-result \
  --request-outputs-file-path ${LOCAL_DIR}/benchmarks/mmlu_outputs.json
```

**NOTE:** To run on all examples from the mmlu test dataset, please change `--num-prompts 14037`

After MMLU benchmark run is completed successfully, you will see logs similar to the following:

```
Using llama-3 tokenizer: /home/user/maxtext/assets/tokenizer_llama3.tiktoken
Loaded 14042 data from mmlu dataset
len(sampled_indices)=1200
In InputRequest, pass in max_output_length: 0 for each sample
The dataset contains 1200 samples.
The filtered dataset contains 963 samples.
Warmup (mode: sampled) is starting.
Benchmarking with a total number of 4 requests
Benchmarking with request rate of 0.0
...
Warmup (mode: sampled) has completed.
Benchmarking with a total number of 963 requests
Benchmarking with request rate of 0.0
...
Mean output size: x
Median output size: x
P99 output size: x
Successful requests: x
Benchmark duration: x s
Total input tokens: x
Total generated tokens: x
Request throughput: x4 requests/s
Input token throughput: xxxx tokens/s
Output token throughput: xx tokens/s
Mean ttft: xxx ms
Median ttft: xxx ms
P99 ttft: xxx ms
Mean ttst: xxx ms
Median ttst: xxx ms
P99 ttst: xxx ms
Mean TPOT: xxx ms
Median TPOT: xxx ms
P99 TPOT: xxx ms
----- Request complete rate time series (window_size = 10 sec) -----
...
----- Output token rate time series (window_size = 10 sec) -----
...
[nltk_data] Downloading package punkt to /home/user/nltk_data...
[nltk_data]   Package punkt is already up-to-date!

Results
```

## Step 8: Run [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) benchmark

### Start MaxText Engine server

To run benchmark, open two SSH sessions connected to the TPU VM. In the first SSH session, start the MaxText Engine server with the following configuration:

```
cd ~
cd maxtext/
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=8192
export MODEL_NAME=llama3.1-70b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=4

# Set XLA flags
export LIBTPU_INIT_ARGS=""" --=false
--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false
--xla_tpu_enable_windowed_einsum_for_all_gather=false
--xla_tpu_prefer_latch_optimized_rhs_layouts=true
--xla_tpu_enable_experimental_fusion_cost_model=false
--xla_tpu_dot_dot_fusion_duplicated=false
--xla_tpu_dot_dot_fusion=true
--xla_jf_conv_input_fusion=true
--xla_jf_conv_output_fusion=true
--xla_tpu_rwb_fusion=false
--xla_tpu_copy_fusion_pad_unpad_ratio=0
--xla_tpu_licm_size_inflation_ratio=1
--xla_tpu_copy_elision_analysis_allowance=150000
--xla_tpu_copy_insertion_use_region_analysis_limit=10000
--xla_tpu_order_dot_after_layout=true
--xla_jf_rematerialization_percent_shared_memory_limit=100
--xla_tpu_use_repeated_instance_for_preferred_prefetch_time=true
--xla_tpu_enforce_prefetch_fifo_order=false
--xla_tpu_prefetch_interval_picker_size_override=6000000
--xla_tpu_async_copy_bandwidth_scaling_factor=1
--xla_tpu_nd_short_transfer_max_chunks=-1
--xla_tpu_enable_aggressive_broadcast_priority_update=true
--xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=SQRT
--xla_tpu_memory_bound_loop_optimizer_options=enabled:true
--xla_tpu_enable_copy_fusion=true
--xla_tpu_enable_cross_program_prefetch_freeing=false
--xla_tpu_enable_dot_strength_reduction=true
--xla_tpu_layout_use_dot_grouping=false
--xla_tpu_msa_inefficient_use_to_copy_ratio=0.5
--xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false
--xla_tpu_vector_load_fusion_window=1024
--xla_tpu_vector_store_fusion_window=256
--xla_jf_conv_reshape_fusion=false
--xla_tpu_input_conv_multi_users=false
--xla_tpu_enable_multi_level_input_dot_dot_fusion=false
--xla_tpu_enable_multi_level_output_dot_dot_fusion=false
--xla_tpu_dot_dot_fusion_separable_convs_only=false
--xla_tpu_enable_multi_level_nested_loop_fusion=true
--xla_tpu_nested_dot_fusion=true
--xla_tpu_enable_multi_level_nested_dot_fusion=false
--xla_jf_enable_multi_output_fusion=true
--xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions=false
--xla_tpu_enable_flash_attention=true
"""

cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${CHECKPOINT_TPU_UNSCANNED} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  attention=dot_product \
  optimize_mesh_for_tpu_v6e=True \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
```

### Run [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) benchmarking

After MaxText Engine server is running, in the second SSH session to the TPU VM, start the MATH500 benchmarking with following configuration:

```
cd ~
source venv-maxtext/bin/activate
cd ~/JetStream
JAX_PLATFORMS=tpu python benchmarks/benchmark_serving.py \
  --model=llama-3 \
  --dataset math500 \
  --num-prompts 500 \
  --request-rate 5 \
  --warmup-mode sampled \
  --max-output-length 8192 \
  --max-output-multiplier=20 \
  --max-input-length=1024 \
  --run-eval True \
  --save-request-outputs \
  --save-result \
  --request-outputs-file-path ${LOCAL_DIR}/benchmarks/math500_outputs.json
```

After the MATH500 benchmark run is completed successfully, you will see logs similar to the following:

```
Using llama-3 tokenizer: /home/user/maxtext/assets/tokenizer_llama3.tiktoken
len(sampled_indices)=500
In InputRequest, pass in max_output_length: 8192 for each sample
The dataset contains 500 samples.
The filtered dataset contains 500 samples.
Warmup (mode: sampled) is starting.
Benchmarking with a total number of 14 requests
Benchmarking with request rate of 5.0
...
Mean output size: xxxx
Median output size: xxxx
P99 output size: xxxx
Successful requests: 500
Benchmark duration: xxx s
Total input tokens: xxxxxx
Total generated tokens: xxxxxx
Request throughput: x requests/s
Input token throughput: xxx tokens/s
Output token throughput: xxx tokens/s
Mean ttft: xxx ms
Median ttft: xxx ms
P99 ttft: xxxx ms
Mean ttst: xxxx ms
Median ttst: xxxx ms
P99 ttst: xxxx ms
Mean TPOT: xx ms
Median TPOT: xx ms
P99 TPOT: xx ms

----- Request complete rate time series (window_size = 10 sec) -----
...
----- Output token rate time series (window_size = 10 sec) -----
...

Results
```

## Cleanup

* Delete v6e TPU VM

```
gcloud compute tpus tpu-vm delete $NODE_ID \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --quiet
```

* Delete Cloud Storage buckets

```
gcloud storage buckets delete ${GCS_BUCKET}
```

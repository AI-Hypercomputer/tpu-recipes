#!/bin/bash

source /app/env.sh

# Extract the number after '-' in TPU_TYPE
TPU_NUM=$(echo "$TPU_TYPE" | grep -oP '(?<=-)\d+')

# Calculate GLOBAL_BATCH_SIZE
GLOBAL_BATCH_SIZE=$(( TPU_NUM * BATCH_PER_DEVICE * NUM_SLICE ))

export GLOBAL_BATCH_SIZE

echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"

export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export ENABLE_PJRT_COMPATIBILITY=true
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=100000
export PROFILE_LOGDIR=${PROFILE_LOG_DIR}
export XLA_PERSISTENT_CACHE_PATH=/app/xla_cache/
export TPU_LIBRARY_PATH=/workspace/_libtpu.so
export NUM_TPU_SLICE=${NUM_SLICE}
export LIBTPU_INIT_ARGS="--xla_tpu_enable_flash_attention=false --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=81920"

huggingface-cli login --token=${HF_TOKEN}

# Note --per_device_train_batch_size is the global batch size since we overwrite the dataloader in the HF trainer.
python3 /workspace/transformers/examples/pytorch/language-modeling/run_clm.py \
--dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 \
--per_device_train_batch_size=${GLOBAL_BATCH_SIZE} --do_train --output_dir=test-clm \
--overwrite_output_dir --config_name=/app/config.json \
--cache_dir=cache --tokenizer_name=mistralai/Mixtral-8x7B-v0.1 \
--block_size=${SEQUENCE_LENGTH} --optim=adafactor --save_strategy=no \
--logging_strategy=no --fsdp="full_shard" \
--fsdp_config=/app/fsdp_config.json --torch_dtype=bfloat16 \
--dataloader_drop_last=yes --max_steps=${MAX_STEP} --gmm --flash_attention
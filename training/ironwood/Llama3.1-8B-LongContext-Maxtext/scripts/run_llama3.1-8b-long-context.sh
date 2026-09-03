# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Running Llama3.1-8B long context (10M) script on TPU Ironwood (v7x)"

# Stop execution if any command exits with error
set -e

# Default workload configs, override by passing KEY=VALUE arguments
export MAX_TARGET_LENGTH=10485760
export ICI_CONTEXT_PARALLELISM=128
export ICI_FSDP_PARALLELISM=1
export PER_DEVICE_BATCH_SIZE=0.0078125

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set up RUN_NAME
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_dvfs_p_state=7 --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_enable_async_all_gather=true --xla_enable_async_collective_permute=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true"
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    model_name=llama3.1-8b steps=8 enable_checkpointing=false \
    override_model_config=true head_dim=256 base_num_query_heads=16 base_num_kv_heads=4 num_vocab_tiling=16 \
    per_device_batch_size=$PER_DEVICE_BATCH_SIZE max_target_length=$MAX_TARGET_LENGTH \
    base_output_directory=$OUTPUT_PATH dataset_type=synthetic packing=false \
    attention=flash use_tokamax_splash=true use_jax_splash=false \
    context_parallel_strategy=ring context_parallel_load_balance=true \
    ici_context_parallelism=$ICI_CONTEXT_PARALLELISM ici_fsdp_parallelism=$ICI_FSDP_PARALLELISM \
    ici_tensor_parallelism=1 remat_policy=custom context=device \
    sa_block_q=2048 sa_block_kv=2048 sa_block_q_dkv=2048 sa_block_kv_dkv=2048 \
    ring_scan_unroll=16 dq_reduction_steps=3

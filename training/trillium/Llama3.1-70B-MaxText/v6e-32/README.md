# Instructions for training Llama3.1-70B-MaxText on TPU trillium (v6e-32)

## XPK setup
Please follow the [XPK_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext

### Install MaxText and Build Docker Image
Please follow the [MAXTEXT_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/MAXTEXT_README.md) to install maxtext and build the docker image. The following variables should be set:

In step 1, use the MaxText [tpu-recipes-v0.1.4](https://github.com/AI-Hypercomputer/maxtext/releases/tag/tpu-recipes-v0.1.4) tag to run this recipe:
```
git checkout tpu-recipes-v0.1.4
```

In step 3, use the jax-stable-stack image containing JAX 0.6.1:
```
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.6.1-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run Maxtext Llama3.1-70B workloads on GKE

### Starting workload

From the MaxText root directory, start your Llama3.1-70B workload.
```
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-32 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama3_1_70b_8192_bs2_bfloat16_no_collective_matmul" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 34, seconds: 17.790, TFLOP/s/device: 443.385, Tokens/s/device: 920.965, total_weights: 524288, loss: 11.457
```

### Workload Details

For reference, here are the `llama3_1_70b_8192_bs2_bfloat16_no_collective_matmul` workload details as found in `MaxText@tpu-recipes-v0.1.4`:

```
MaxTextModel(
    model_name="llama3_1-70b-8192-bs2-bfloat16-no-collective-matmul",
    model_type="llama3.1-70b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "enable_checkpointing": False,
        "sa_block_q": 2048,
        "sa_block_kv": 2048,
        "sa_block_kv_compute": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 2048,
        "sa_block_q_dq": 2048,
        "sa_block_kv_dq": 2048,
        "sa_use_fused_bwd_kernel": True,
        "profiler": "xplane",
        "skip_first_n_steps_for_profiler": 10,
        "profiler_steps": 5,
        "weight_dtype": "bfloat16",
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.DISABLE_COLLECTIVE_MATMUL
    ),
),
```

This equivalent workload code can be found in the [maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/9f1820b472ef362e7b5c782fe1d6fda8a0943eff/benchmarks/maxtext_trillium_model_configs.py) file within the MaxText repository.
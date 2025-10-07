# Instructions for training Llama4-Maverick-17B-128E Maxtext on TPU v5p-256

This documents present steps to run Llama4-Maverick-17B-128E [MaxText](https://github.com/google/maxtext) workload through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

## XPK setup

Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK.

## Prep for Maxtext

Please follow the [MAXTEXT_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md) to install maxtext and build the docker image. The following variables should be set:

In step 1, Use the MaxText [tpu-recipes-v0.1.3](https://github.com/AI-Hypercomputer/maxtext/releases/tag/tpu-recipes-v0.1.3) tag to run this recipe:
```
git checkout tpu-recipes-v0.1.3
```

In step 3, use the jax-stable-stack image containing JAX 0.5.2:
```
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run workloads

From the MaxText root directory, start your workload

```
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v5p-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama4_maverick_dropless_v5p_256" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:

```
completed step: 12, seconds: 24.792, TFLOP/s/device: 160.005, Tokens/s/device: 1321.725, total_weights: 4194304, loss: 10.034
```

Workload details can be found in `MaxText@tpu-recipes-v0.1.3` [here](https://github.com/AI-Hypercomputer/maxtext/blob/9ca35d7e60b71303b9f6fa885447d32e8a612c47/benchmarks/maxtext_v5p_model_configs.py#L151-L196):

```
    MaxTextModel(
        model_name="llama4_maverick_dropless_v5p_256",
        model_type="llama4-17b-128e",
        tuning_params={
            "per_device_batch_size": 4,
            "max_target_length": 8192,
            "ici_fsdp_parallelism": 32,
            "ici_tensor_parallelism": 4,
            "enable_checkpointing": False,
            "dtype": "bfloat16",
            "weight_dtype": "float32",
            "megablox": True,
            "sparse_matmul": True,
            "dataset_type": "synthetic",
            "opt_type": "adamw",
            "skip_first_n_steps_for_profiler": 5,
            "profiler_steps": 3,
            "profiler": "xplane",
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "out_proj": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "reuse_example_batch": 1,
            "sa_block_q": 2048,
            "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048,
            "tokenizer_path": "meta-llama/Llama-4-Maverick-17B-128E",
        },
        xla_flags=(
            xla_flags_library.MOE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
    )
```

# Instructions for training Gemma3-12B-MaxText on TPU trillium (2 slices of v6e-256)

## XPK setup
Please follow the [XPK_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext

### Install MaxText and Build Docker Image
Please follow the [MAXTEXT_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md) to install maxtext and build the docker image. The following variables should be set:

In step 1, use the MaxText [tpu-recipes-v0.1.5](https://github.com/AI-Hypercomputer/maxtext/releases/tag/tpu-recipes-v0.1.5) tag to run this recipe:
```
git checkout tpu-recipes-v0.1.5
```

In step 3, use:
```
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable JAX_VERSION=0.7.0
```

## Run Maxtext Gemma3-12B workloads on GKE

### Starting workload

From the MaxText root directory, start your Gemma3-12B workload.
```
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=2  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="gemma3_12b_32768_2x_v6e256" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 29, seconds: 7.793, TFLOP/s/device: 328.139, Tokens/s/device: 4204.799, total_weights: 16777216, loss: 11.151
```

### Workload Details

For reference, here are the `gemma3_12b_32768_2x_v6e256` workload details as found in `MaxText@tpu-recipes-v0.1.5`:

```
MaxTextModel(
    model_name="gemma3-12b-32768-2x-v6e256",
    model_type="gemma3-12b",
    tuning_params={
        "per_device_batch_size": 1,
        "num_vocab_tiling": 16,
        "ici_fsdp_parallelism": 1,
        "ici_fsdp_transpose_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "device",
        "query_proj": "remat",
        "key_proj": "remat",
        "value_proj": "remat",
        "max_target_length": 32768,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "skip_first_n_steps_for_profiler": 10,
        "profiler_steps": 2,
        "tokenizer_path": os.path.join("assets", "tokenizer.gemma3"),
        "sa_block_q": 1024,
        "sa_block_kv": 1024,
        "sa_block_kv_compute": 1024,
        "sa_block_q_dkv": 512,
        "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 512,
        "sa_block_q_dq": 1024,
        "sa_block_kv_dq": 1024,
    },
    xla_flags=(
        xla_flags_library.CUSTOM_VMEM_LIMIT_FLAG(vmem_limit=122880)
    ),
)
```

This equivalent workload code can be found in the [maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/50bafeb98299458f73d853b1325787a6d241d10c/benchmarks/maxtext_trillium_model_configs.py) file within the MaxText repository.
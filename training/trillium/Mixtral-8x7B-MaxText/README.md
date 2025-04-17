# Instructions for training Mixtral-8x7B-MaxText on TPU trillium

## XPK setup
Please follow the [XPK_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 

### Install MaxText and Build Docker Image
Please follow the [MAXTEXT_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/MAXTEXT_README.md) to install maxtext and build the docker image. The following variables should be set:

In step 1, use the MaxText [tpu-recipes-v0.1.2](https://github.com/AI-Hypercomputer/maxtext/releases/tag/tpu-recipes-v0.1.2) tag to run this recipe:
```
git checkout tpu-recipes-v0.1.2
```

In step 3, use the jax-stable-stack image containing JAX 0.5.2:
```
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run Maxtext Mixtral-8x7B workloads on GKE

### Starting workload

From the MaxText root directory, start your Mixtral workload.

```
python3 -m benchmarks.benchmark_runner xpk \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="mixtral_8x7b_dropped" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 11, seconds: 13.484, TFLOP/s/device: 302.311, Tokens/s/device: 3645.203, total_weights: 12582912, loss: 10.546
```

### Workload Details

For reference, here are the `mixtral_8x7b_dropped` workload details as found in `MaxText@tpu-recipes-v0.1.2`:

```
MaxTextModel(
    model_name="mixtral_8x7b_dropped",
    model_type="mixtral-8x7b",
    tuning_params={
        "per_device_batch_size": 12,
        "ici_fsdp_parallelism": -1,
        "max_target_length": 4096,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "out_proj": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "megablox": False,
        "sparse_matmul": False,
        "capacity_factor": 1.25,
        "tokenizer_path": "assets/tokenizer.mistral-v1",
    },
    xla_flags=(
        xla_flags_library.MOE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)
```

This equivalent workload code can be found in the [maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/tpu-recipes-v0.1.2/benchmarks/maxtext_trillium_model_configs.py) file within the MaxText repository.

# Instructions for training Llama3.1-405B-MaxText on TPU trillium

## XPK setup
Please follow the [XPK_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext

### Install MaxText and Build Docker Image
Please follow the [MAXTEXT_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md) to install maxtext and build the docker image. The following variables should be set:

In step 1, use the MaxText [tpu-recipes-v0.1.2](https://github.com/AI-Hypercomputer/maxtext/releases/tag/tpu-recipes-v0.1.2) tag to run this recipe:
```
git checkout tpu-recipes-v0.1.2
```

In step 3, use the jax-stable-stack image containing JAX 0.5.2:
```
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run Maxtext Llama3.1-405B workloads on GKE

### Starting workload

From the MaxText root directory, start your Llama3.1-405B workload.
```
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=2  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama3_1_405b_8192_pure_fsdp_ici" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 14, seconds: 54.803, TFLOP/s/device: 392.454, Tokens/s/device: 149.482, total_weights: 4194304, loss: 0.297
```

### Workload Details

For reference, here are the `llama3_1_405b_8192_pure_fsdp_ici` workload details as found in `MaxText@tpu-recipes-v0.1.2`:

```
MaxTextModel(
    model_name="llama3-1-405b-8192-pure-fsdp-ici",
    model_type="llama3.1-405b",
    tuning_params={
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": 256,
        "dcn_fsdp_parallelism": 2,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
    ),
)
```

This equivalent workload code can be found in the [maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/tpu-recipes-v0.1.2/benchmarks/maxtext_trillium_model_configs.py) file within the MaxText repository.
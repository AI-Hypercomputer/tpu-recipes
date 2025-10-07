# Instructions for training GPT3-175B-Maxtext on TPU trillium

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

## Run Maxtext GPT3-175B workloads on GKE

### Starting workload

From the MaxText root directory, start your GPT3-175B workload
```
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME}  \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="gpt_3_175b_bf16" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 15, seconds: 17.182, TFLOP/s/device: 384.891, Tokens/s/device: 357.580, total_weights: 1572864, loss: 388.622
```

### Workload Details

For reference, here are the `gpt_3_175b_bf16` workload details as found in `MaxText@tpu-recipes-v0.1.2`:

```
MaxTextModel(
    model_name="gpt-3-175b-bf16",
    model_type="gpt3-175b",
    tuning_params={
        "per_device_batch_size": 3,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "attention": "flash",
        "gcs_metrics": True,
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
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.DISABLE_BUNDLE_AWARE_COST_MODEL
    ),
)
```

This equivalent workload code can be found in the [maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/tpu-recipes-v0.1.2/benchmarks/maxtext_trillium_model_configs.py) file within the MaxText repository.
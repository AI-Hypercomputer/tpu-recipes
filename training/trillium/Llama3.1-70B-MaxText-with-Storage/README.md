# Instructions for training Llama3.1-70B-MaxText on TPU trillium (v6e-256) with Google Cloud Storage (GCS)

## GCS Bucket setup
1. Create two buckets: one to hold the dataset and one to use for checkpoints. To create regional HNS buckets use the following commands:
```
# Set variables
export DATASET_BUCKET="dataloading-bucket-name"
export CHECKPOINT_BUCKET="checkpoint-bucket-name"
export REGION="us-central1"

# Create dataset bucket
gcloud storage buckets create gs://${DATASET_BUCKET} --location=${REGION}  --default-storage-class=Standard --enable-hierarchical-namespace --uniform-bucket-level-access

# Create checkpoint bucket  
gcloud storage buckets create gs://${CHECKPOINT_BUCKET} --location=${REGION}  --default-storage-class=Standard --enable-hierarchical-namespace --uniform-bucket-level-access
```
Replace the following values:  
- `<DATASET_BUCKET>`:the name of your Cloud Storage bucket with training dataset. Do not include the gs:// prefix  
- `<CHECKPOINT_BUCKET>`: the name of your Cloud Storage bucket where checkpoints will be written. Do not include the gs:// prefix
- `<REGION>`: the region where your GKE cluster is located ([available locations](https://cloud.google.com/storage/docs/locations#location-r))

2. Follow these [instructions](https://github.com/AI-Hypercomputer/maxtext/blob/b93beba652db6b3f4e6c82dc48a83b03229f5d3a/getting_started/Data_Input_Pipeline.md#tfds-pipeline) to download the Allenai c4 dataset to the dataset bucket.
Then follow these [instructions](https://github.com/google/array_record/tree/main/beam) to convert the dataset into ArrayRecord.

## XPK setup
1. Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK.
2. GCSFuse lets you mount and access Cloud Storage buckets as local file systems, so applications can read and write objects in your bucket using standard file system semantics. You'll need to use the below commands to create [XPK storage resources](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#storage) for both the dataset and checkpoint buckets in order to mount them to the MaxText workload using GCSFuse. For the dataset bucket and checkpoint bucket use separate manifest files `checkpoint_pvc.yaml` and `dataset_pvc.yaml` from this repo.
Be sure to update `volumeHandle` in the yamls with your correct bucket names. Creating a bucket and xpk storage is a one time setup.
```
export RECIPE_REPO="path-to-this-recipe-repo" # Update

cd ~/xpk

python3 xpk.py storage attach dataset-bucket --type=gcsfuse --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mountpoint=/tmp/dataset --readonly=false --bucket=$DATASET_BUCKET --size=64 --automount=false --manifest=$RECIPE_REPO/tpu-recipes/training/trillium/Llama3.1-70B-MaxText-with-Storage/dataset_pvc.yaml

python3 xpk.py storage attach checkpoint-bucket --type=gcsfuse --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mountpoint=/tmp/ckpt --readonly=false --bucket=$CHECKPOINT_BUCKET --size=64 --automount=false --manifest=$RECIPE_REPO/tpu-recipes/training/trillium/Llama3.1-70B-MaxText-with-Storage/checkpoint_pvc.yaml
```


## Prep for MaxText

### Install MaxText and Build Docker Image
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/MAXTEXT_README.md) to install maxtext and build the docker image.

In step 2, use the jax-stable-stack image containing JAX 0.5.2:
```
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run MaxText Llama3.1-70B workloads on GKE

### Starting workload

From the MaxText root directory, start your Llama3.1-70B workload.

Run MaxText Llama 3.1 70B with synthetic data and no checkpointing:
```
python3 benchmarks/benchmark_runner.py xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=$CLUSTER \
    --base_output_directory=$OUTPUT_DIR \
    --model_name="llama3_1_70b_8192_synthetic" \
    --num_steps=100 \
    --base_docker_image=maxtext_base_image
```

Run MaxText Llama 3.1 70B with checkpointing and loading real data from GCS:
```
python3 benchmarks/benchmark_runner.py xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER} \
    --base_output_directory=/tmp/ckpt \
    --model_name="llama3_1_70b_8192_rd_ckpt_grain" \
    --num_steps=100 \
    --base_docker_image=maxtext_base_image \
    --xpk_storage=dataset-bucket --xpk_storage=checkpoint-bucket
```

If you would like to run on multiple slices of v6e-256, you may modify the `--num_slices` flag.

### Workload Details

For reference, here are the `llama3_1_70b_8192_synthetic` and `llama3_1_70b_8192_rd_ckpt_grain` workload details:

```
  MaxTextModel(
        model_name="llama3_1-70b-8192",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 4,
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
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
    )


    MaxTextModel(
        model_name="llama3_1_70b_8192_rd_ckpt_grain",
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
            "dataset_path": "/tmp/dataset",
            "dataset_type": "grain",
            "grain_train_files": "/tmp/dataset/array-record/c4/en/3.0.1/c4-train.array_record*",
            "grain_worker_count": 24,
            "enable_checkpointing": True,
            "async_checkpointing": True,
            "checkpoint_period": 20,
            "sa_block_q": 2048,
            "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True,
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE 
            +  " --xla_tpu_iova_dma_chunk_size_bytes=104857"
        ),
  )
```

This equivalent workload code can be found in the [maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/1e4d513ad70dd4074d975a9f7936295008d4b900/benchmarks/maxtext_trillium_model_configs.py#L1103-L1146) file within the MaxText repository.

## Clean-up
You can run the following commands to detach the XPK storage resources (this removes the PersistentVolumes and PersistentVolumeClaims created by the `xpk storage attach` commands from your GKE cluster).
```
# Detach dataset storage
python3 xpk.py storage detach dataset-bucket \
  --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE

# Detach checkpoint storage
python3 xpk.py storage detach checkpoint-bucket \
  --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE
```


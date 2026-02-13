# Instructions for training Stable Diffusion 2 on TPU v5p

This documents present steps to run StableDiffusion [MaxDiffusion](https://github.com/google/maxdiffusion/tree/main/src/maxdiffusion) workload through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

Setup XPK and create cluster [XPK Userguide](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md)

Build a local docker image.

```
LOCAL_IMAGE_NAME=maxdiffusion_base_image
docker build  --no-cache --network host -f ./maxdiffusion.Dockerfile -t ${LOCAL_IMAGE_NAME} .
```

Run workload using xpk.

```
export BASE_OUTPUT_DIR=gs://output_bucket/
DATA_DIR=gs://jfacevedo-maxdiffusion/laion400m/raw_data/tf_records_512_encoder_state_fp32
COMMITS=eac9132ef8b1a977372e29720fabc478529cd364
NUM_SLICES=1

xpk workload create \
--cluster <cluster_name> \
--base-docker-image maxdiffusion_base_image \
--workload ${USER}-sd21-v5p \
--tpu-type=<tpu_type> \
--num-slices=${NUM_SLICES}  \
--command "bash run_v5p-ddp-pbs-16.sh DATA_DIR=${DATA_DIR} BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR} COMMITS=${COMMITS} NUM_SLICES=${NUM_SLICES} "
```

MFU Calculation.

Above only Unet is trainable module, from FLOPS count, Per Step FLOPS = 2.41G FLOPS @BS=1, we get the MFU
```
MFU = Per Step FLOPS * BatchSize Per Device / Step Time / Per Device Peak FLOPS
```

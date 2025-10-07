# Instructions for training Stable Diffusion XL on TPU v5p

This documents present steps to run StableDiffusion [MaxDiffusion](https://github.com/google/maxdiffusion/tree/main/src/maxdiffusion) workload through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

Setup XPK and create cluster [XPK Userguide](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md)

Build a local docker image.

```
LOCAL_IMAGE_NAME=maxdiffusion_base_image
docker build  --no-cache --network host -f ./docker/maxdiffusion.Dockerfile -t ${LOCAL_IMAGE_NAME} .
```

Run workload using xpk.

```
export BASE_OUTPUT_DIR=gs://output_bucket/
export NUM_SLICES=1

xpk workload create \
--cluster <cluster_name> \
--base-docker-image maxdiffusion_base_image \
--workload ${USER}-sdxl-v5p \
--tpu-type=<tpu_type> \
--num-slices=${NUM_SLICES}  \
--zone $ZONE \
--project $PROJECT \
--command "bash scripts/run_v5p-ddp-pbs-1.sh BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR} COMMITS=00150750841e9155669fd1ac4c6f2fcd0e0654e0"
```

MFU Calculation.

Above only UNET is trainable model, FLOPS count = 162.27 TFLOPS @BS=8, we get the MFU
```
MFU = UNET FLOPS / Step Time / Per Device Peak FLOPS
```
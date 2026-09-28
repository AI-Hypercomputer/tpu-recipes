# Instructions for training Stable Diffusion XL on TPU v5p

This documents present steps to run StableDiffusion [MaxDiffusion](https://github.com/google/maxdiffusion/tree/main/src/maxdiffusion) workload through [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit) (`gcluster`) tool.

Setup Cluster Toolkit (`gcluster` v1.104.0) and create cluster [Cluster Toolkit Cloud TPU deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview). `run_recipe.sh` prints the `gcluster` install commands if `gcluster` is not on your `PATH`.

Build a local docker image and push it to an Artifact Registry Docker repository (`REPOSITORY`).

```
cd tpu-recipes/training/v5p/SDXL-MaxDiffusion
LOCAL_IMAGE_NAME=maxdiffusion_base_image
docker build  --no-cache --network host -f ./docker/maxdiffusion.Dockerfile -t ${LOCAL_IMAGE_NAME} .
export WORKLOAD_IMAGE=us-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${LOCAL_IMAGE_NAME}
docker tag ${LOCAL_IMAGE_NAME} ${WORKLOAD_IMAGE}
docker push ${WORKLOAD_IMAGE}
```

Run workload using Cluster Toolkit. The script runs on a TPU v5p slice with topology `4x4x4` (64 chips).

```
export BASE_OUTPUT_DIR=gs://output_bucket
export PROJECT_ID=$PROJECT
export CLUSTER_NAME=$CLUSTER_NAME
export ZONE=$ZONE

bash cluster_toolkit/run_recipe.sh
```

To delete the workload, set `WORKLOAD_NAME` to the name printed by `run_recipe.sh` and run:

```
gcluster job cancel ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

MFU Calculation.

Above only UNET is trainable model, FLOPS count = 162.27 TFLOPS @BS=8, we get the MFU
```
MFU = UNET FLOPS / Step Time / Per Device Peak FLOPS
```
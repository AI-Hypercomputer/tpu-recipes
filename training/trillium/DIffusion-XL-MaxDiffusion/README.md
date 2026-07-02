# Instructions for training MaxDiffusion SDXL on TPU trillium

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxdiffusion 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/training/trillium/MAXDIFFUSION_README.md) to install maxdiffusion and build docker image

Download pretrained stable_xl_base from [huggingface](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main)
##### `gsutil -m cp stable-diffusion-xl-base-1.0 ${OUTPUT_DIR}/checkpoints `

Prepare dataset and store at local folder

`python -m unittest input_pipeline_interface_test.InputPipelineInterface.test_make_pokemon_iterator_sdxl_cache`

Upload prepared dataset to gcs location

`gsutil -m cp /tmp/pokemon-gpt4-captions_xl ${OUTPUT_DIR}/dataset `
## Run Maxdiffusion SDXL workloads on GKE

### Test Env
jaxlib=0.4.35

[maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion.git)@269b6216ac65adb9e7044ec454879dc99856d5e9

### Starting workload

From the maxdiffusion root directory, start your SDXL workload on v6e-256

```
python3 ~/xpk/xpk.py  workload create --cluster  $CLUSTER_NAME  --workload $USER-maxdiffusion --command "bash sdxl-v6e-256-pbds-1.sh $OUT_DIR"  \
--base-docker-image=maxdiffusion_base_image \
--tpu-type=v6e-256 --num-slices=1 --zone=$ZONE --project=$PROJECT_ID
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 254, seconds: 0.164, TFLOP/s/device: 123.764, loss: 0.055
```

start your SDXL workload on multi-slices of v6e-256

```
python3 ~/xpk/xpk.py  workload create --cluster  $CLUSTER_NAME  --workload $USER-maxdiffusion --command "bash sdxl-2xv6e-256-pbds-1.sh $OUT_DIR"  \
--base-docker-image=maxdiffusion_base_image \
--tpu-type=v6e-256 --num-slices=2 --zone=$ZONE --project=$PROJECT_ID
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 92, seconds: 0.228, TFLOP/s/device: 89.120, loss: 0.057
```
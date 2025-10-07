# Instructions for training GPT3-175B-Maxtext on TPU trillium

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md) to install maxtext and build docker image

## Run Maxtext GPT3-175B workloads on GKE

### Test Env
jaxlib=0.4.35

libtpu-nightly=20241028

[maxtext](https://github.com/AI-Hypercomputer/maxtext.git)@e7292a3a572792a0d797fc8977b21d0f255729f1

### Starting workload

From the MaxText root directory, start your GPT3-175B workload

```
python3 benchmarks/benchmark_runner.py --project=${PROJECT} --zone={zone} --device_type=v6e-256 --num_slices=1  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="gpt_3_175b" --libtpu_version=20241028 --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
step: 100, seconds: 14.245, TFLOP/s/device: 464.261, Tokens/s/device: 431.318, total_weights: 1572864, loss: 0.000
```
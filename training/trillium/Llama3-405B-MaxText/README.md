# Instructions for training Llama3.1-405B-MaxText on TPU trillium

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/MAXTEXT_README.md) to install maxtext and build docker image

## Run Maxtext Llama3.1-405B workloads on GKE

### Test Env
jaxlib=0.4.36

libtpu-nightly=20241106

[maxtext](https://github.com/AI-Hypercomputer/maxtext.git)@5c45ca3869bd62543031d9858c61811e04d6b147

### Starting workload

From the MaxText root directory, start your Llama3.1-405B workload.
```
python3 benchmarks/benchmark_runner.py --project=$PROJECT --zone=$ZONE --device_type=v6e-256 --num_slices=2  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="llama3_1_405b_8192_fsdp_dcn" --libtpu_version=20241028 --base_docker_image maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 5, seconds: 58.805, TFLOP/s/device: 365.740, Tokens/s/device: 139.307, total_weights: 4194304, loss: 0.000
```
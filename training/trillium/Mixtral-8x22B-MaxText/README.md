# Instructions for training Mixtral-8x22B-MaxText on TPU trillium

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md) to install maxtext and build docker image

## Run Maxtext Mixtral-8x7B workloads on GKE

### Test Env
jaxlib=0.4.35

libtpu-nightly=20241119

[maxtext](https://github.com/AI-Hypercomputer/maxtext.git)@261a8be0fc5e909ef9da0521df62549e650ebb79

### Starting workload

From the MaxText root directory, start your Mixtral workload.

Bf16 run:
```
python3 benchmarks/benchmark_runner.py --project=${PROJECT} --zone={zone} --device_type=v6e-256 --num_slices=1  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="mixtral_8x22b_dropped" --libtpu_version=20241119 --base_docker_image=maxtext_base_image
```

Note: After commit `f64c51a2d8c115e98b6c4d24d90b546e5f0f826e`, use the xpk flag when running the benchmark script. For example: `python3 benchmarks/benchmark_runner.py xpk --project=${PROJECT} ...`.

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 9, seconds: 24.706, TFLOP/s/device: 332.463, Tokens/s/device: 1326.307, total_weights: 8388608, loss: 0.045
```
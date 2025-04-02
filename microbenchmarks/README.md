# Microbenchmarks

## Setup

Set up a v6e TPU VM:
```
gcloud compute tpus tpu-vm create ${TPU_NAME} /
        --project ${PROJECT_ID} /
        --zone=${ZONE} /
        --accelerator-type=v6e-1 /
        --version=v2-alpha-tpuv6e
```
If needed, see the full list of [available zones](https://cloud.google.com/tpu/docs/regions-zones).

SSH into the VM:
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

More info on the previous commands can be found in the [Google Cloud documentation](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm).

Clone the repo and install the dependencies:
```bash
git clone https://github.com/AI-Hypercomputer/tpu-recipes.git
cd tpu-recipes/microbenchmarks
pip install -r requirements.txt
```

## Run Matmul Benchmark

Usage example:
```
python benchmark_matmul.py \
  --dim 4096 4096 4096 \
  --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 \
  --matcher="jit_matmul.*"
```

Example output:
```
dtype: bfloat16, matrix Dimensions: (4096, 4096, 4096), time taken (median): 0.16358503900000002 ms, TFLOPS: 840.1682348958574
```

Run `python benchmark_matmul.py -h` to view the how to set the arguments.


## HBM Bandwidth Benchmark

Usage example:
```
python benchmark_hbm.py \
  --num_elements=16777216 \
  --matcher="jit_my_copy.*"
```

Example output:
```
Tensor size (bytes): 33554432, time taken (ms, median): 0.049359414, bandwidth (GBps, median): 1359.5960438266143
```

Run `python benchmark_hbm.py -h` to view the how to set the arguments.

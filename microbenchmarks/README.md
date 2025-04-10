# Microbenchmarks

## Setup

Set up a v6e TPU VM for single-chip microbenchmarks:
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
  --dim 8192 8192 8192 \
  --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 \
  --trace_matcher="jit_matmul.*"
```

Example output:
```
dtype: bfloat16, matrix dimensions: (8192, 8192, 8192), time taken (median, ms): 1.328756094, TFLOPS: 827.474382048629
```

The figure below shows the trace of the example above. Setting
 `--trace_matcher="jit_matmul.*"` means that the completion time is measured by
 the duration of the compiled [`matmul`](benchmark_matmul.py#L19) function on
 TPUs, which excludes the communication overheads between the host (CPU) and
 TPUs.


![Trace Image](https://services.google.com/fh/files/misc/trace.png)


If `--trace_matcher` is not set, the completion time will be measured by timing
 the function on the host, which includes the compilation and communication
 overheads, including kernel launch, data transfer, synchronization, etc..

Example:
```
python benchmark_matmul.py \
  --dim 8192 8192 8192 \
  --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536
```

Output:

```
dtype: bfloat16, matrix dimensions: (8192, 8192, 8192), time taken (median, ms): 1.457810401916504, TFLOPS: 754.2212803054033
```

Run `python benchmark_matmul.py -h` to view the how to set the other arguments.

## HBM Bandwidth Benchmark

Usage example:
```
python benchmark_hbm.py \
  --num_elements=16777216 \
  --trace_matcher="jit_my_copy.*"
```

Example output:
```
Tensor size (bytes): 33554432, time taken (ms, median): 0.049359414, bandwidth (GBps, median): 1359.5960438266143
```

Run `python benchmark_hbm.py -h` to view the how to set the arguments.

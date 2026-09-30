# Instructions for running Collectives Benchmark on TPU trillium (v6e-256)

## Cluster Toolkit setup
Please follow the [Cluster Toolkit Cloud TPU deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview) to create your GKE cluster with Cluster Toolkit (`gcluster`) v1.104.0. `run_recipe.sh` prints the `gcluster` install commands if `gcluster` is not on your `PATH`.

This benchmark expects the TCP receive buffer sizes `net.ipv4.tcp_rmem = 4096 41943040 314572800` on the TPU nodes. Set it on the v6e node pool with the `gke-node-pool` module's `linux_node_config` setting in your cluster blueprint:
```yaml
linux_node_config:
  sysctls:
    net.ipv4.tcp_rmem: "4096 41943040 314572800"
```
The workload prints the value in effect (`net.ipv4.tcp_rmem: ...`) at the start of its logs.

## Run Collectives on v6e-256

### Starting workload

Launch the Cluster Toolkit workload, example to run on 1 slice of v6e-256:
```
cd tpu-recipes/microbenchmarks/trillium/collectives/cluster_toolkit
export PROJECT_ID=$PROJECT
export CLUSTER_NAME=$CLUSTER_NAME
export ZONE=$ZONE
export BASE_OUTPUT_DIR=gs://your-gcs-bucket
./run_recipe.sh
```

To run on more than 1 slice, set `NUM_SLICES` to the target number of slices (`2` or `4`). The script uses the corresponding yaml config file, e.g. `configs/2x_v6e_256.yaml`:
```
NUM_SLICES=2 ./run_recipe.sh
```

From your workload logs, you should start seeing benchmark logs:
```
psum_dcn: Matrix size: 17408x17408, dtype=<class 'jax.numpy.bfloat16'>, matrix_size_gbyte=0.606076928,achieved_bandwidth_gbyte_s=4.1130934137328214
psum_ici: Matrix size: 17408x17408, dtype=<class 'jax.numpy.bfloat16'>, matrix_size_gbyte=0.606076928,achieved_bandwidth_gbyte_s=235.7595345022845
```

Results will be printed out and also stored at `/tmp/microbenchmarks/collectives`. When the benchmark finishes, each worker uploads its log to `${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}/logs/` and its stored results to `${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}/results/`.

To delete the workload, set `WORKLOAD_NAME` to the name printed by `run_recipe.sh` and run:
```
gcluster job cancel ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```

### Run with a custom yaml config
If you would like to run with a custom defined yaml with modified configurations (e.g. warmup_tries, tries, matrix_dim_range) you may do so by uploading it to a GCS bucket and pointing `run_recipe.sh` at it; the workload pulls the yaml file from GCS and references it in the benchmark command.

Start by creating a yaml file `your_config.yaml`. Take a look at [1x_v6e_256.yaml](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/blob/35c10a42e8cfab7593157327dd3ad3150e4c001d/configs/1x_v6e_256.yaml) for an example yaml config. Then upload it to your GCS bucket:
```
gcloud storage cp your_config.yaml gs://your-gcs-bucket/your_config.yaml
```

Then run with `GCS_CONFIG_URI` set to the uploaded file:
```
GCS_CONFIG_URI=gs://your-gcs-bucket/your_config.yaml ./run_recipe.sh
```

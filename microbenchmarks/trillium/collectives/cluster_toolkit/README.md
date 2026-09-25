# Instructions for running Collectives Benchmark on TPU trillium (v6e-256)

## Cluster Toolkit setup
Please follow the [Cluster Toolkit Cloud TPU deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview) to create your GKE cluster with Cluster Toolkit (`gcluster`) v1.104.0.

## Run Collectives on v6e-256

### Starting workload (Cluster Toolkit)

Launch the Cluster Toolkit workload using `run_recipe.sh` (for example, on 1 slice of v6e-256):
```bash
cd tpu-recipes/microbenchmarks/trillium/collectives/cluster_toolkit
export PROJECT_ID=${PROJECT}
export CLUSTER_NAME=${CLUSTER_NAME}
export ZONE=${ZONE}
export NUM_SLICES=1
./run_recipe.sh
```

To run on more than 1 slice (e.g. `2` or `4` slices of `v6e-256`), set `NUM_SLICES` before running `./run_recipe.sh`. `run_recipe.sh` automatically selects the matching configuration file (`configs/${NUM_SLICES}x_v6e_256.yaml`) inside `accelerator-microbenchmarks`:
```bash
export NUM_SLICES=2
./run_recipe.sh
```

From your workload logs, you should start seeing metrics outputs like the following:
```
...
psum_dcn: Matrix size: 17408x17408, dtype=<class 'jax.numpy.bfloat16'>, matrix_size_gbyte=0.606076928,achieved_bandwidth_gbyte_s=24.160443188732568
psum_ici: Matrix size: 17408x17408, dtype=<class 'jax.numpy.bfloat16'>, matrix_size_gbyte=0.606076928,achieved_bandwidth_gbyte_s=235.7595345022845
```

Results will be printed out and also stored at `/tmp/microbenchmarks/collectives` inside the container. To automatically save the stored results to a GCS bucket when the benchmark finishes, set `BASE_OUTPUT_DIR`:
```bash
export BASE_OUTPUT_DIR=gs://<your-gcs-bucket>
./run_recipe.sh
```

### Custom Configurations
You can view the configuration parameters in the [accelerator-microbenchmarks repo](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks/tree/trillium-collectives/configs).
To run with a custom configuration, create a YAML file locally and upload it to a GCS bucket:

```bash
gcloud storage cp your_config.yaml gs://<your-gcs-bucket>/your_config.yaml
```

Then set `GCS_CONFIG_URI` and run `./run_recipe.sh`:
```bash
export GCS_CONFIG_URI=gs://<your-gcs-bucket>/your_config.yaml
export NUM_SLICES=1
./run_recipe.sh
```

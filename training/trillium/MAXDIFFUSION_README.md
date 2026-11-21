# Prep for Maxdiffusion workloads on GKE
1. Clone [maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion.git) repo and move to its directory
```
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion
git checkout ${maxdiffusion_HASH}
```

2. Run the following commands to build the docker image
```
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.4.35-rev1
```

3. Upload your docker image to Container Registry
```
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_runner
```

4. Create your GCS bucket
```
OUTPUT_DIR=gs://v6e-demo-run #<your_GCS_folder_for_results>
gcloud storage buckets create ${OUTPUT_DIR}  --project ${PROJECT}
```

5. Specify your workload configs
```
export PROJECT=#<your_compute_project>
export ZONE=#<your_compute_zone>
export CLUSTER_NAME=v6e-demo #<your_cluster_name>
export OUTPUT_DIR=gs://v6e-demo/ #<your_GCS_folder_for_results>
```
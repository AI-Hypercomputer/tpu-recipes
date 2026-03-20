# Prep for MaxText workloads on GKE

> **_NOTE:_** We recommend using Python 3.12 for MaxText workloads, as it is our primary supported version. Other Python versions may encounter compatibility issues.

1. Clone [MaxText](https://github.com/google/maxtext) repo and move to its directory
```shell
git clone https://github.com/google/maxtext.git
cd maxtext

# Specify MaxText commit id or tag you want to checkout
# You can find MaxText tag here: https://github.com/AI-Hypercomputer/maxtext/tags
MAXTEXT_COMMIT_ID_OR_TAG=#<commit_id_or_tag> e.g. tpu-recipes-v0.1.2
git checkout ${MAXTEXT_COMMIT_ID_OR_TAG}
```

2. Install MaxText dependencies

   **Optional:** Use a virtual environment to setup and run your workloads. This can help with errors like `This environment is externally managed`:
```shell
# Install uv, a fast Python package installer
pip install uv

# Create a virtual environment
VENV_NAME=#<your_virtual_env_name>
uv venv --python 3.12 --seed $VENV_NAME
source $VENV_NAME/bin/activate

# Install MaxText and its dependencies
uv pip install -e .[tpu] --resolution=lowest
install_maxtext_github_deps
```

3. Run the following commands to build the docker image
```shell
# Example BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
BASE_IMAGE=#<stable_stack_image_with_desired_jax_version>
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

4. Upload your docker image to Container Registry
```shell
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_runner
```

5. Create your GCS bucket
```shell
GCS_PROJECT=#<your_GCS_project>
OUTPUT_DIR=#<your_GCS_bucket_for_results> e.g. gs://v6e-demo-run
gcloud storage buckets create ${OUTPUT_DIR}  --project ${GCS_PROJECT}
```

6. Specify your workload configs
```shell
export PROJECT=#<your_compute_project>
export ZONE=#<your_compute_zone>
export CLUSTER_NAME=#<your_cluster_name>
```

# FAQ

1. If you see the following error when creating your virtual environment in step 2, install the 
required dependency using the output's provided command. You may need to run the command with `sudo`. This 
example is for Python3.10.
```
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.10-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.

Failing command: /home/bvandermoon/venvp3/bin/python3

-bash: /home/bvandermoon/venvp3/bin/activate: No such file or directory
```

2. If you see an error like the following while building your Docker image, there could be a pip versioning
conflict in your cache.

```
ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE. If you have updated the
package versions, please update the hashes. Otherwise, examine the package contents carefully;
someone may have tampered with them.
     unknown package:
         Expected sha256 b3e54983cd51875855da7c68ec05c05cf8bb08df361b1d5b69e05e40b0c9bd62
              Got        f3b7ea1da59dc4f182437cebc7ef37b847d55c7ebfbc3ba286302f1c89ff5929
```

Try deleting your pip cache file: `rm ~/.cache/pip -rf`. Then retry the Docker build

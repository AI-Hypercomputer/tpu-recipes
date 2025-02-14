# Instructions for training Llama 3 70B on Trillium TPU

This user guide provides a concise overview of the essential steps required to
run Hugging Face (HF) Llama 3 70B training on Trillium TPUs.

## Environment Setup

Please follow the corresponding TPU generation's user guide to setup the GCE TPUs
first.

Please replace all your-* with your TPUs' information.

```
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
```

You may use this command to create a 256 chip Trillium pod:

```bash
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v6e --topology 16x16 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

## Steps to Run HF Llama 3 70B

The following setup runs the training job with Llama 3 70B on GCE TPUs using
the docker image from this registry
(`us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-xla/llama3-70b:feb14build`).
The docker image uses torch and torch_xla nightly build from 02/11/2024
and comes with all the package dependency needed to run the model training.
All the command below should run from your own machine (not the TPU host you
created). The dockerfile used is to build the image is at https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/Llama3-70B-PyTorch/GCE/tpu.Dockerfile

1. git clone and navigate to this README repo and run training script:

```bash
git clone --depth 1 https://github.com/AI-Hypercomputer/tpu-recipes.git
cd training/trillium/Llama3-70B-PyTorch/GCE
```

2. Edit `env.sh` to add the hugging face token and/or setup the training parameters.

```bash
# add your hugging face token
HF_TOKEN=hf_***
```

3. Run the training script:

```bash
./benchmark.sh
```

`benchmark.sh` script will: upload 1) environment parameters in `env.sh`, 2)
model related config in `config.json`, `fsdp_config.json`, 3) docker launch
script in `host.sh` and 4) python training command in `train.sh` into all TPU
workers, and starts the training afterwards. When all training steps complete,
it will print out training metrics of each worker as below in terminal:

```
[worker :0] ***** train metrics *****
[worker :0]   epoch                    =        0.3125
[worker :0]   total_flos               = 10915247040GF
[worker :0]   train_loss               =         9.278
[worker :0]   train_runtime            =    0:46:45.60
[worker :0]   train_samples            =         32816
[worker :0]   train_samples_per_second =          3.65
[worker :0]   train_steps_per_second   =         0.007
```

In addition,  it will copy back the trained model under `output/*`.

### Appendix:
- historical docker image releases: `us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-xla/llama3-70b:nightly-sep28`

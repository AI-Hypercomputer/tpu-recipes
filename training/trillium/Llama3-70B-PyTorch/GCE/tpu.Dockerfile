# Base package containing nightly PyTorch/XLA
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_cxx11_20250211

# Install transformers library
ARG TRANSFORMERS_REPO=https://github.com/pytorch-tpu/transformers.git
ARG TRANSFORMERS_REF=flash_attention_minibatch_v6e
WORKDIR /workspace
RUN git clone "${TRANSFORMERS_REPO}" transformers && cd transformers && git checkout "${TRANSFORMERS_REF}"

# Install transformers dependencies
WORKDIR /workspace/transformers
RUN pip3 install git+file://$PWD accelerate datasets protobuf evaluate "huggingface_hub[cli]" \
    "torch_xla[pallas]" \
    -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
    -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

WORKDIR /workspace

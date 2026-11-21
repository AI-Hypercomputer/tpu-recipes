FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla@sha256:6a98314f2375b493d999871535be6a0a4317691be51534dec81f38e38684e2f6

# Set the working directory
WORKDIR /workspace

# Install PyTorch and Torchvision with the nightly builds
RUN pip3 install --pre torch==2.7.0.dev20250121+cpu torchvision==0.22.0.dev20250121+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
# Install torch_xla for TPU usage
RUN pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.7.0.dev20250121+cxx11-cp310-cp310-linux_x86_64.whl

RUN pip3 install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Clone the PyTorch TPU transformers repository
RUN git clone --branch piz/hybridmesh https://github.com/pytorch-tpu/transformers.git


WORKDIR /workspace/transformers

RUN pip3 install -e .

WORKDIR /workspace

# Set the default command (optional, for example, to run Python)
CMD ["/bin/bash"]

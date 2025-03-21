# Setup

## Step 0: (optional) Create a virtual environment for Python packages to install

```bash
export WORKDIR=$(pwd)  # set current dir as workdir (can set to something else)
cd $WORKDIR
sudo apt install python3.10-venv
python -m venv venv
source venv/bin/activate
```

## Step 1: Get JetStream-PyTorch github repository

```bash
git clone https://github.com/google/jetstream-pytorch.git
cd jetstream-pytorch/
git checkout jetstream-v0.2.4
```

## Step 2: Setup JetStream and JetStream-PyTorch
```bash
source install_everything.sh
pip install -U --pre jax jaxlib libtpu-nightly requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```


Do not install jetstream separately, the above command will install everything.

## Step 2.1: Make sure there is a working version of Jax that can access TPUs:

```bash
python -c "import jax; print(jax.devices())"
```

Should print out something like this:

```bash
(venv) hanq@t1v-n-9c8a4ce2-w-0:/run/user/2003/jetstream-pytorch$ python -c "import jax; print(jax.devices())"
[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=0, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=0, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,3,0), core_on_chip=0)]
```


## Step 3: Run jetstream pytorch

List out supported models

```bash
jpt list
```

This will print out list of support models and variants:

```bash
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-13b-hf
meta-llama/Llama-2-70b-hf
meta-llama/Llama-2-70b-chat-hf
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3-8B-Instruct
meta-llama/Meta-Llama-3-70B
meta-llama/Meta-Llama-3-70B-Instruct
google/gemma-2b
google/gemma-2b-it
google/gemma-7b
google/gemma-7b-it
mistralai/Mixtral-8x7B-v0.1
mistralai/Mixtral-8x7B-Instruct-v0.1
```

Note: Before you run the following command for the first time, make sure you
authenticate with HuggingFace. 

To run jetstream-pytorch server with one model:
```bash
jpt serve --model_id meta-llama/Llama-2-7b-chat-hf
```

The first time you run this model, the `jpt serve` command will attempt
to download weights from HuggingFace which requires that you authenticate with
HuggingFace. 

To authenticate, run `huggingface-cli login` to set your access token, or pass 
your HuggingFace access token to the `jpt serve` command using the `--hf_token` 
flag:

```bash
jpt serve --model_id meta-llama/Llama-2-7b-chat-hf --hf_token=...
```

For more information about HuggingFace access tokens, see [Access Tokens](https://huggingface.co/docs/hub/en/security-tokens). 

To log in using [HuggingFace Hub](https://huggingface.co/docs/hub/en/index), 
run the following command and follow the prompts:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

After the weights are downloaded, you no longer need to specify the `--hf_token`
flag.

To run this model with `int8` quantization, add `--quantize_weights=1`.
Quantization will be done on the flight as the weight loads.

Weights downloaded from HuggingFace are stored by default in a directory called
`checkpoints` folder in the directory where you run `jpt`. You can change also
specify a directory using the `--working_dir` flag.

If you wish to use your own checkpoint, place them inside the 
`checkpoints/<org>/<model>/hf_original` dir (or the corresponding subdir in 
`--working_dir`). For example, Llama2-7b checkpoints will be in `checkpoints/meta-llama/Llama-2-7b-hf/hf_original/*.safetensors`. You can replace these files with modified weights in 
HuggingFace format. 

## send one gPRC

Jetstream-pytorch uses gPRC for handling requests, the script below demonstrates how to
send gRPC in Python. You can also use other gPRC clients.

```python
import requests
import os
import grpc

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc

prompt = "What are the top 5 languages?"

channel = grpc.insecure_channel("localhost:9000")
stub = jetstream_pb2_grpc.OrchestratorStub(channel)

request = jetstream_pb2.DecodeRequest(
    text_content=jetstream_pb2.DecodeRequest.TextContent(
        text=prompt
    ),
    priority=0,
    max_tokens=2000,
)

response = stub.Decode(request)
output = []
for resp in response:
  output.extend(resp.stream_content.samples[0].text)

text_output = "".join(output)
print(f"Prompt: {prompt}")
print(f"Response: {text_output}")
```

# Benchmark

In terminal tab 1, start the server:
```bash
jpt serve --model_id meta-llama/Llama-2-7b-chat-hf
```

In terminal tab 2, run the benchmark:
One time setup
```bash
source venv/bin/activate
cd jetstream-pytorch/deps/JetStream/benchmarks
pip install -r requirements.in
```

Run the benchmark
```bash
export model_name=llama-2
export tokenizer_path=../../../checkpoints/meta-llama/Llama-2-7b-chat-hf/hf_original/tokenizer.model
python benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 1000  --dataset openorca --save-request-outputs --warmup-mode=sampled --model=$model_name
```

See more at https://github.com/google/JetStream-pytorch

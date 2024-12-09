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

To run jetstream-pytorch server with one model:
```bash
jpt serve --model_id meta-llama/Llama-2-7b-chat-hf
```
If it's the first time you run this model, it will download weights from 
HuggingFace. 

HuggingFace's Llama3 weights are gated, so you need to either run 
`huggingface-cli login` to set your token, OR, pass your hf_token explicitly.

To pass hf token explicitly, add `--hf_token` flag
```bash
jpt serve --model_id meta-llama/Llama-2-7b-chat-hf --hf_token=...
```

To login using huggingface hub, run:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
Then follow its prompt.

After the weights are downloaded,
Next time when you run this `--hf_token` will no longer be required.

To run this model in `int8` quantization, add `--quantize_weights=1`.
Quantization will be done on the flight as the weight loads.

Weights downloaded from HuggingFace will be stored by default in `checkpoints` folder.
in the place where `jpt` is executed.

You can change where the weights are stored with `--working_dir` flag.

If you wish to use your own checkpoint, then, place them inside 
of the `checkpoints/<org>/<model>/hf_original` dir (or the corresponding subdir in `--working_dir`). For example,
Llama2-7b checkpoints will be at `checkpoints/meta-llama/Llama-2-7b-hf/hf_original/*.safetensors`. You can replace these files with modified
weights in HuggingFace format. 

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
import functools
import re

path = '/workspace/tpu_inference/tpu_inference/models/jax/gemma4_mtp.py'
with open(path, 'r') as f:
    text = f.read()

# 1. Patch the imports
import_target = "from tpu_inference.layers.jax.rope_interface import apply_rope"
import_replacement = "from tpu_inference.layers.jax.rope_interface import apply_rope\nfrom tpu_inference.layers.common.quantization import quantize_kv"
if import_target in text:
    text = text.replace(import_target, import_replacement)
    print("Imports patched successfully!")
else:
    print("Warning: import target not found!")

# 2. Patch __call__ signature
target_block = """    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,"""
replacement_block = """    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        hidden_states: Optional[jax.Array] = None,
        attention_metadata: Optional[AttentionMetadata] = None,"""
if target_block in text:
    text = text.replace(target_block, replacement_block)
    print("Call signature patched successfully!")
else:
    print("Warning: call signature target not found!")

# 3. Patch __call__ body (hidden_states initialization)
body_target = '        layer_name_to_kv_cache = ('
body_replacement = """        if hidden_states is None:
            backbone_hidden_size = getattr(self.vllm_config.speculative_config.draft_model_config.hf_config, 'backbone_hidden_size', 5376)
            draft_hidden_size = self.vllm_config.speculative_config.draft_model_config.hf_config.text_config.hidden_size
            hidden_size = 2 * backbone_hidden_size - draft_hidden_size
            dtype = self.vllm_config.model_config.dtype
            if input_ids.ndim == 1:
                seq_len = input_ids.shape[0]
                hidden_states = jnp.zeros((seq_len, hidden_size), dtype=dtype)
            else:
                batch_size, seq_len = input_ids.shape
                hidden_states = jnp.zeros((batch_size, seq_len, hidden_size), dtype=dtype)
        layer_name_to_kv_cache = ("""
if body_target in text:
    text = text.replace(body_target, body_replacement)
    print("Call body patched successfully!")
else:
    print("Warning: call body target not found!")

# 4. Patch Gemma4MTPAttention.__init__ (kv_cache_quantized_dtype setup)
init_target = """        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, ))
            if config.attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self.is_kv_shared_layer = True"""
init_replacement = """        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, ))
            if config.attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self.is_kv_shared_layer = True

        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)"""
if init_target in text:
    text = text.replace(init_target, init_replacement)
    print("Gemma4MTPAttention.__init__ patched successfully!")
else:
    print("Warning: Gemma4MTPAttention.__init__ target not found!")

# 5. Patch Gemma4MTPAttention.__call__ (quantize dummy_k/dummy_v)
call_target = """        num_tokens = q.shape[0]
        dummy_k = jnp.zeros((num_tokens, self.num_kv_heads, self.head_dim),
                            dtype=q.dtype)
        dummy_v = jnp.zeros((num_tokens, self.num_kv_heads, self.head_dim),
                            dtype=q.dtype)

        new_kv_cache, outputs = attention("""
call_replacement = """        num_tokens = q.shape[0]
        dummy_k = jnp.zeros((num_tokens, self.num_kv_heads, self.head_dim),
                            dtype=q.dtype)
        dummy_v = jnp.zeros((num_tokens, self.num_kv_heads, self.head_dim),
                            dtype=q.dtype)

        if self.kv_cache_quantized_dtype:
            dummy_k, dummy_v = quantize_kv(self.kv_cache_quantized_dtype, dummy_k, dummy_v)

        new_kv_cache, outputs = attention("""
if call_target in text:
    text = text.replace(call_target, call_replacement)
    print("Gemma4MTPAttention.__call__ patched successfully!")
else:
    print("Warning: Gemma4MTPAttention.__call__ target not found!")

with open(path, 'w') as f:
    f.write(text)
print("gemma4_mtp.py successfully patched!")

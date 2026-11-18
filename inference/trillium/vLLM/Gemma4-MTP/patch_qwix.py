path = '/workspace/tpu_inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py'
with open(path, 'r') as f:
    text = f.read()

target_block = """    head_sizes = kv_cache_head_size
    if isinstance(head_sizes, int):
        head_sizes = (head_sizes, ) * num_hidden_layers

    num_kv_heads_tuple = kv_cache_num_kv_heads
    if isinstance(num_kv_heads_tuple, int):
        num_kv_heads_tuple = (num_kv_heads_tuple, ) * num_hidden_layers"""

replacement_block = """    head_sizes = kv_cache_head_size
    num_kv_heads_tuple = kv_cache_num_kv_heads
    default_head_size = head_sizes if isinstance(head_sizes, int) else head_sizes[0]
    default_num_kv_heads = num_kv_heads_tuple if isinstance(num_kv_heads_tuple, int) else num_kv_heads_tuple[0]
    dynamic_head_sizes = [default_head_size] * num_hidden_layers
    dynamic_num_kv_heads = [default_num_kv_heads] * num_hidden_layers
    inner_model = model
    if hasattr(model, 'model'):
        inner_model = model.model
    if hasattr(inner_model, 'layers'):
        for idx, layer in enumerate(inner_model.layers):
            if idx >= num_hidden_layers:
                break
            attn = getattr(layer, 'self_attn', getattr(layer, 'attn', None))
            if attn is not None:
                if hasattr(attn, 'num_kv_heads'):
                    dynamic_num_kv_heads[idx] = attn.num_kv_heads
                if hasattr(attn, 'head_dim'):
                    dynamic_head_sizes[idx] = attn.head_dim
        head_sizes = tuple(dynamic_head_sizes)
        num_kv_heads_tuple = tuple(dynamic_num_kv_heads)
    else:
        if isinstance(head_sizes, int):
            head_sizes = (head_sizes, ) * num_hidden_layers
        if isinstance(num_kv_heads_tuple, int):
            num_kv_heads_tuple = (num_kv_heads_tuple, ) * num_hidden_layers"""

if target_block in text:
    text = text.replace(target_block, replacement_block)
    with open(path, 'w') as f:
        f.write(text)
    print('qwix_utils.py successfully patched for heterogeneous KV cache!')
else:
    print('Error: qwix_utils target block not found!')

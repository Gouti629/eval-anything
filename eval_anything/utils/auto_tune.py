from __future__ import annotations

_STANDARD_BATCH_SIZES = [8, 16, 32, 48, 64, 72, 96, 128]
_STANDARD_TOKEN_SIZES = [512, 1024, 2048, 3072, 4096, 8192]
_SAFETY_FACTOR = 0.6


def _snap_down(value: float, standards: list[int]) -> int | None:
    best = None
    for s in standards:
        if s <= value:
            best = s
    return best


def _get_model_params(model, backend: str) -> tuple[int, int, int]:
    if backend == "vllm":
        cfg = model.llm_engine.model_config.hf_config
    else:
        cfg = model.config

    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", 32)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", 4096)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    head_dim = hidden_size // num_heads if num_heads else 128
    kv_dim = num_kv_heads * head_dim if num_kv_heads else hidden_size

    dtype = None
    if backend == "vllm":
        try:
            dtype = model.llm_engine.model_config.dtype
        except Exception:
            pass
    else:
        dtype = getattr(model, "dtype", None)

    import torch
    bytes_map = {
        torch.float16: 2, torch.bfloat16: 2,
        torch.float32: 4, torch.float64: 8, torch.int8: 1,
    }
    dtype_bytes = bytes_map.get(dtype, 2)
    return num_layers, kv_dim, dtype_bytes


def auto_tune_params(
    model,
    backend: str,
    batch_size: int | None = None,
    max_new_tokens: int | None = None,
    estimated_input_len: int = 256,
) -> tuple[int, int]:
    """Compute batch_size or max_new_tokens from available GPU memory.

    - Both given: returned as-is.
    - batch_size given, max_new_tokens=None: compute largest standard token count that fits.
    - max_new_tokens given, batch_size=None: compute largest standard batch size that fits.
    - Both None: pick max_new_tokens=2048, compute batch size.

    Falls back to (batch_size or 8, max_new_tokens or 512) when no CUDA device is available.
    """
    if batch_size is not None and max_new_tokens is not None:
        return batch_size, max_new_tokens

    import torch
    if not torch.cuda.is_available():
        return batch_size or 8, max_new_tokens or 512

    free_mem, _total = torch.cuda.mem_get_info()
    usable = int(free_mem * _SAFETY_FACTOR)

    num_layers, hidden_size, dtype_bytes = _get_model_params(model, backend)
    kv_per_token = 2 * num_layers * hidden_size * dtype_bytes

    if batch_size is not None and max_new_tokens is None:
        budget_tokens = usable / (batch_size * kv_per_token)
        available_output = budget_tokens - estimated_input_len
        result = _snap_down(available_output, _STANDARD_TOKEN_SIZES)
        computed = result if result is not None else 512
        print(f"  Auto-tuned max_new_tokens={computed} "
              f"(batch_size={batch_size}, free={free_mem // (1024**2)}MB, "
              f"kv/tok={kv_per_token}B)")
        return batch_size, computed

    if max_new_tokens is not None and batch_size is None:
        seq_tokens = estimated_input_len + max_new_tokens
        max_batch = usable / (seq_tokens * kv_per_token)
        result = _snap_down(max_batch, _STANDARD_BATCH_SIZES)
        computed = result if result is not None else 8
        print(f"  Auto-tuned batch_size={computed} "
              f"(max_new_tokens={max_new_tokens}, free={free_mem // (1024**2)}MB, "
              f"kv/tok={kv_per_token}B)")
        return computed, max_new_tokens

    max_new_tokens = 2048
    seq_tokens = estimated_input_len + max_new_tokens
    max_batch = usable / (seq_tokens * kv_per_token)
    result = _snap_down(max_batch, _STANDARD_BATCH_SIZES)
    computed = result if result is not None else 8
    print(f"  Auto-tuned batch_size={computed}, max_new_tokens={max_new_tokens} "
          f"(free={free_mem // (1024**2)}MB, kv/tok={kv_per_token}B)")
    return computed, max_new_tokens

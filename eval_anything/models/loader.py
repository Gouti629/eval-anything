_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "auto": "auto",
}


def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
):
    from vllm import LLM

    if not device.startswith("cuda"):
        raise ValueError(f"vLLM requires a CUDA device, got {device!r}")

    vllm_dtype = _DTYPE_MAP.get(dtype)
    if vllm_dtype is None:
        raise ValueError(f"Unsupported dtype {dtype!r}. Use: {list(_DTYPE_MAP.keys())}")

    llm = LLM(
        model=model_path,
        dtype=vllm_dtype,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer

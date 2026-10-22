def load_model_hf(model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = torch_dtype_map.get(dtype)
    if torch_dtype is None:
        raise ValueError(
            f"Unsupported dtype {dtype!r}. Use: {list(torch_dtype_map.keys())}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if device.startswith("cuda"):
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":")[1])
            except ValueError:
                pass
        load_kwargs["device_map"] = {"": idx}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()

    return model, tokenizer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    torch_dtype = _DTYPE_MAP.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype {dtype!r}. Use: {list(_DTYPE_MAP.keys())}")

    if torch_dtype == torch.bfloat16 and device != "cpu":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print(f"Warning: bfloat16 not supported on this GPU, falling back to float16")
            torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

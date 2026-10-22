from __future__ import annotations

from eval_anything.utils.generation import GenerationResult


def batched_generate_hf(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    num_return_sequences: int = 1,
) -> list[GenerationResult]:
    import torch

    model_max = getattr(model.config, "max_position_embeddings", None) or tokenizer.model_max_length
    max_input_len = model_max - max_new_tokens

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_len,
        )
    finally:
        tokenizer.padding_side = orig_padding_side

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[:, input_len:]
    texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    pad_id = tokenizer.pad_token_id
    results: list[GenerationResult] = []
    for i in range(0, len(texts), num_return_sequences):
        batch_texts = texts[i:i + num_return_sequences]
        batch_ids = new_ids[i:i + num_return_sequences]
        if pad_id is not None:
            token_count = sum(int((ids != pad_id).sum()) for ids in batch_ids)
            truncated = any(
                int((ids != pad_id).sum()) >= max_new_tokens for ids in batch_ids
            )
        else:
            token_count = sum(len(ids) for ids in batch_ids)
            truncated = any(len(ids) >= max_new_tokens for ids in batch_ids)
        results.append(GenerationResult(
            text=batch_texts,
            token_count=token_count,
            truncated=truncated,
        ))
    return results

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import StoppingCriteria


@dataclass
class GenerationResult:
    text: list[str]
    token_count: int
    truncated: bool


class TimeBudget(StoppingCriteria):
    def __init__(self, seconds: int):
        self.deadline = time.monotonic() + seconds
        self.expired = False

    def __call__(self, input_ids, scores, **kwargs):
        if time.monotonic() > self.deadline:
            self.expired = True
            return True
        return False


def _is_truncated(token_ids, eos_token_id, max_new_tokens: int) -> bool:
    n_generated = token_ids.shape[-1]
    if n_generated < max_new_tokens:
        return False
    if eos_token_id is not None and n_generated > 0:
        last = token_ids[..., -1].item() if token_ids.dim() == 1 else None
        if last == eos_token_id:
            return False
    return True


def batched_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 8,
    timeout: int = 120,
    num_return_sequences: int = 1,
    use_cache: bool = True,
) -> list[GenerationResult]:
    all_outputs: list[GenerationResult] = []

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": use_cache,
    }

    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    if num_return_sequences > 1:
        gen_kwargs["num_return_sequences"] = num_return_sequences

    eos_id = tokenizer.eos_token_id

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        timer = TimeBudget(timeout)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                stopping_criteria=[timer],
                **gen_kwargs,
            )

        timed_out = timer.expired

        input_len = encoded.input_ids.shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        if num_return_sequences > 1:
            for j in range(len(batch_prompts)):
                start = j * num_return_sequences
                end = start + num_return_sequences
                seq_tokens = new_tokens[start:end]
                n_tok = sum(
                    (seq_tokens[k] != tokenizer.pad_token_id).sum().item()
                    for k in range(seq_tokens.shape[0])
                )
                trunc = timed_out or any(
                    _is_truncated(seq_tokens[k], eos_id, max_new_tokens)
                    for k in range(seq_tokens.shape[0])
                )
                all_outputs.append(GenerationResult(
                    text=decoded[start:end],
                    token_count=n_tok,
                    truncated=trunc,
                ))
        else:
            for k, text in enumerate(decoded):
                row_tokens = new_tokens[k]
                n_tok = (row_tokens != tokenizer.pad_token_id).sum().item()
                trunc = timed_out or _is_truncated(row_tokens, eos_id, max_new_tokens)
                all_outputs.append(GenerationResult(
                    text=[text],
                    token_count=n_tok,
                    truncated=trunc,
                ))

    return all_outputs

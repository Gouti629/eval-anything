from __future__ import annotations

import threading

import torch


def _generate_with_timeout(model, input_ids, attention_mask, gen_kwargs, timeout: int):
    result = [None]
    error = [None]

    def _run():
        try:
            with torch.inference_mode():
                result[0] = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Thread cannot be killed in Python; mark as timed out.
        # The daemon thread will be cleaned up on process exit.
        # The GPU memory from this generation will persist until the
        # next cuda.empty_cache() or model deletion.
        raise TimeoutError(
            f"Generation timed out after {timeout}s. "
            f"The background thread may still be running — "
            f"consider reducing max_new_tokens or batch_size."
        )
    if error[0] is not None:
        raise error[0]
    return result[0]


def batched_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 8,
    timeout: int = 120,
    num_return_sequences: int = 1,
) -> list[list[str]]:
    all_outputs: list[list[str]] = []

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    if num_return_sequences > 1:
        gen_kwargs["num_return_sequences"] = num_return_sequences

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        output_ids = _generate_with_timeout(
            model,
            encoded.input_ids,
            encoded.attention_mask,
            gen_kwargs,
            timeout=timeout,
        )

        input_len = encoded.input_ids.shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        if num_return_sequences > 1:
            for j in range(len(batch_prompts)):
                start = j * num_return_sequences
                end = start + num_return_sequences
                all_outputs.append(decoded[start:end])
        else:
            for text in decoded:
                all_outputs.append([text])

    return all_outputs

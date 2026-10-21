from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: list[str]
    token_count: int
    truncated: bool


def batched_generate(
    llm,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    num_return_sequences: int = 1,
) -> list[GenerationResult]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=num_return_sequences,
    )

    outputs = llm.generate(prompts, sampling_params)

    results: list[GenerationResult] = []
    for output in outputs:
        texts = [o.text for o in output.outputs]
        token_count = sum(len(o.token_ids) for o in output.outputs)
        truncated = any(o.finish_reason == "length" for o in output.outputs)
        results.append(GenerationResult(
            text=texts,
            token_count=token_count,
            truncated=truncated,
        ))
    return results

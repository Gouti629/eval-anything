from datasets import load_dataset

from eval_harness.rewards.scorers import SCORERS


class GenericEval:
    def __init__(self, config: dict):
        self.name = config["name"]
        self.dataset_name = config["dataset"]
        self.dataset_config = config.get("config", None)
        self.split = config.get("split", "test")
        self.question_field = config["question_field"]
        self.answer_field = config["answer_field"]
        self.system_prompt = config.get("system_prompt", "")
        self.scorer_name = config.get("scorer", "exact_match")
        self.formatter_name = config.get("formatter", "plain")
        self.filter_expr = config.get("filter", None)
        self.delimiter = config.get("delimiter", None)
        self.choices_field = config.get("choices_field", None)
        self.prepend_prompt = config.get("prepend_prompt", False)
        self._scorer_fn = SCORERS[self.scorer_name]

    @property
    def is_code_eval(self) -> bool:
        return self.scorer_name == "code_exec"

    @property
    def metric_name(self) -> str:
        return "pass@1" if self.is_code_eval else "accuracy"

    def load_data(self) -> list[dict]:
        from datasets import concatenate_datasets

        kwargs = {"split": self.split}
        configs = self.dataset_config
        if configs is None:
            configs = [None]
        elif isinstance(configs, str):
            configs = [configs]

        datasets = []
        for cfg in configs:
            if cfg is not None:
                ds = load_dataset(self.dataset_name, cfg, **kwargs)
            else:
                ds = load_dataset(self.dataset_name, **kwargs)
            datasets.append(ds)

        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = concatenate_datasets(datasets)

        if self.filter_expr:
            combined = combined.filter(lambda row: eval(self.filter_expr, {}, row))

        results = []
        for row in combined:
            question = self._get_field(row, self.question_field)
            answer = self._get_field(row, self.answer_field)
            entry = {
                "question": question,
                "ground_truth": answer,
                "_raw": {k: v for k, v in row.items()},
            }
            results.append(entry)
        return results

    def _get_field(self, row: dict, field: str):
        if "." in field:
            parts = field.split(".")
            val = row
            for p in parts:
                val = val[p]
            return val
        return row[field]

    def format_prompt(self, example: dict, tokenizer) -> str:
        question = example["question"]

        if self.formatter_name == "multiple_choice" and self.choices_field:
            choices = example["_raw"][self.choices_field]
            question = question + "\n\n" + self._format_choices(choices)
        elif self.formatter_name == "code_completion":
            pass

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n")
        return prompt

    def _format_choices(self, choices) -> str:
        if isinstance(choices, dict) and "label" in choices and "text" in choices:
            return "\n".join(
                f"{label}. {text}"
                for label, text in zip(choices["label"], choices["text"])
            )
        if isinstance(choices, list):
            labels = "ABCDEFGHIJKLMNOP"
            return "\n".join(
                f"{labels[i]}. {c}" for i, c in enumerate(choices)
            )
        return str(choices)

    def score(self, generated: str, ground_truth, example: dict | None = None) -> float:
        kwargs = {}
        if self.delimiter:
            kwargs["delimiter"] = self.delimiter
        if self.prepend_prompt and example is not None:
            kwargs["prepend"] = example["question"]
        return self._scorer_fn(generated, ground_truth if isinstance(ground_truth, str) else str(ground_truth), **kwargs)

    def extract_answer(self, generated: str) -> str | None:
        from eval_harness.rewards.scorers import _strip_think
        clean = _strip_think(generated)
        if self.scorer_name == "number_extract":
            if self.delimiter and self.delimiter in clean:
                return clean.split(self.delimiter)[-1].strip()
            import re
            numbers = re.findall(r"-?\d+\.?\d*", clean)
            return numbers[-1] if numbers else None
        if self.scorer_name == "boxed_latex":
            from eval_harness.rewards.scorers import _extract_boxed
            return _extract_boxed(clean)
        if self.scorer_name == "letter_match":
            import re
            match = re.search(r"[A-D]", clean.upper())
            return match.group(0) if match else None
        if self.scorer_name == "code_exec":
            from eval_harness.rewards.scorers import _extract_code
            return _extract_code(clean)
        return clean.strip()[:100] if clean else None

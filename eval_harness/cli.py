import argparse
import sys

from eval_harness.runner import evaluate


def main():
    parser = argparse.ArgumentParser(
        prog="eval-harness",
        description="Lightweight LLM evaluation harness",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="HuggingFace model name or local checkpoint path (repeat for multiple)",
    )

    eval_group = parser.add_argument_group("eval specification (use --eval_config OR inline flags)")
    eval_group.add_argument("--eval_config", type=str, default=None,
                            help="Path to YAML eval config, or 'default' for built-in benchmarks")
    eval_group.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    eval_group.add_argument("--config", type=str, default=None, help="HuggingFace dataset config name")
    eval_group.add_argument("--split", type=str, default="test", help="Dataset split")
    eval_group.add_argument("--question_field", type=str, default=None, help="Field name for question/prompt")
    eval_group.add_argument("--answer_field", type=str, default=None, help="Field name for ground truth")
    eval_group.add_argument("--scorer", type=str, default="exact_match",
                            help="Scoring method (exact_match, number_extract, boxed_latex, letter_match, code_exec)")
    eval_group.add_argument("--delimiter", type=str, default=None, help="Delimiter for number_extract scorer")
    eval_group.add_argument("--formatter", type=str, default="plain",
                            help="Prompt formatter (plain, multiple_choice, code_completion)")
    eval_group.add_argument("--choices_field", type=str, default=None, help="Field name for multiple choice options")
    eval_group.add_argument("--filter", type=str, default=None, help="Filter expression for dataset rows")
    eval_group.add_argument("--system_prompt", type=str, default="", help="System prompt string")
    eval_group.add_argument("--prepend_prompt", action="store_true", help="Prepend question to output before scoring")
    eval_group.add_argument("--eval_name", type=str, default=None, help="Name for inline eval (defaults to dataset name)")

    parser.add_argument("--comment", type=str, default=None, help="Free-text comment for logging")
    parser.add_argument("--output", type=str, default="./results", help="Results directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation token limit")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (0.0 = greedy)")
    parser.add_argument("--pass_k", type=int, default=None, help="Number of samples for pass@k (code evals only)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")

    args = parser.parse_args()

    temperature = args.temperature
    if args.pass_k is not None and temperature == 0.0:
        temperature = 0.8

    eval_configs = _resolve_eval_configs(args)
    if not eval_configs:
        print("Error: specify --eval_config or --dataset with --question_field and --answer_field", file=sys.stderr)
        sys.exit(1)

    evaluate(
        models=args.model,
        eval_configs=eval_configs,
        comment=args.comment,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        pass_k=args.pass_k,
        device=args.device,
    )


def _resolve_eval_configs(args) -> list[dict]:
    if args.eval_config:
        return _load_eval_config(args.eval_config)

    if args.dataset and args.question_field and args.answer_field:
        name = args.eval_name or args.dataset.split("/")[-1]
        config = {
            "name": name,
            "dataset": args.dataset,
            "split": args.split,
            "question_field": args.question_field,
            "answer_field": args.answer_field,
            "scorer": args.scorer,
            "formatter": args.formatter,
            "system_prompt": args.system_prompt,
        }
        if args.config:
            config["config"] = args.config
        if args.delimiter:
            config["delimiter"] = args.delimiter
        if args.choices_field:
            config["choices_field"] = args.choices_field
        if args.filter:
            config["filter"] = args.filter
        if args.prepend_prompt:
            config["prepend_prompt"] = True
        return [config]

    return []


def _load_eval_config(path: str) -> list[dict]:
    import yaml
    import importlib.resources

    if path == "default":
        ref = importlib.resources.files("eval_harness").joinpath("default_evals.yaml")
        text = ref.read_text(encoding="utf-8")
    else:
        with open(path) as f:
            text = f.read()

    configs = yaml.safe_load(text)
    if not isinstance(configs, list):
        raise ValueError("eval config YAML must be a list of eval definitions")
    return configs


if __name__ == "__main__":
    main()

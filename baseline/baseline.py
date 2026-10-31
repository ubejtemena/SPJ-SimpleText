"""Reproducible BART baselines for Cochrane text simplification."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "facebook/bart-base"


def default_data_file(task: str, split: str) -> Path:
    if task == "sentence":
        return ROOT / "data" / f"cochraneauto_sents_{split}.csv"
    return ROOT / "data" / f"cochrane_docs_{split}.csv"


def read_records(path: Path, task: str) -> list[dict[str, str]]:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"complex", "simple"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: complex, simple")
        for line_number, row in enumerate(reader, start=2):
            source = (row.get("complex") or "").strip()
            if not source:
                raise ValueError(f"{path}:{line_number} has an empty complex input")
            if task == "sentence":
                try:
                    targets = ast.literal_eval(row["simple"])
                except (SyntaxError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line_number} has an invalid sentence target list"
                    ) from error
                if not isinstance(targets, list) or not all(
                    isinstance(target, str) for target in targets
                ):
                    raise ValueError(
                        f"{path}:{line_number} target must be a list of strings"
                    )
                target = " ".join(item.strip() for item in targets if item.strip())
            else:
                target = (row.get("simple") or "").strip()
            if not target:
                if task != "sentence":
                    raise ValueError(f"{path}:{line_number} has an empty reference")

            record = {
                "source": source,
                "reference": target,
                "pair_id": row.get("pair_id", ""),
            }
            for column in ("para_id", "sent_id"):
                if column in row:
                    record[column] = row[column]
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def sari_score(source: str, prediction: str, reference: str, max_order: int = 4) -> float:
    """Return sentence-level SARI on the 0-100 scale (single reference)."""
    source_tokens = source.lower().split()
    prediction_tokens = prediction.lower().split()
    reference_tokens = reference.lower().split()
    component_scores = {"add": [], "keep": [], "delete": []}

    for order in range(1, max_order + 1):
        source_ngrams = _ngrams(source_tokens, order)
        prediction_ngrams = _ngrams(prediction_tokens, order)
        reference_ngrams = _ngrams(reference_tokens, order)
        source_set = set(source_ngrams)
        prediction_set = set(prediction_ngrams)
        reference_set = set(reference_ngrams)

        system_keep = prediction_set & source_set
        reference_keep = source_set & reference_set
        good_keep = system_keep & reference_keep
        keep_precision = len(good_keep) / len(system_keep) if system_keep else 0.0
        keep_recall = len(good_keep) / len(reference_keep) if reference_keep else 0.0
        component_scores["keep"].append(_f1(keep_precision, keep_recall))

        system_add = prediction_set - source_set
        reference_add = reference_set - source_set
        good_add = system_add & reference_add
        add_precision = len(good_add) / len(system_add) if system_add else 0.0
        add_recall = len(good_add) / len(reference_add) if reference_add else 0.0
        component_scores["add"].append(_f1(add_precision, add_recall))

        system_delete = source_set - prediction_set
        reference_delete = source_set - reference_set
        good_delete = system_delete & reference_delete
        delete_precision = (
            len(good_delete) / len(system_delete) if system_delete else 0.0
        )
        component_scores["delete"].append(delete_precision)

    return 100.0 * statistics.mean(
        statistics.mean(scores) for scores in component_scores.values()
    )


def calculate_metrics(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    metrics: list[str],
    device: str,
) -> dict[str, float]:
    results: dict[str, float] = {}
    if "sari" in metrics:
        results["sari"] = statistics.mean(
            sari_score(source, prediction, reference)
            for source, prediction, reference in zip(sources, predictions, references)
        )
    if "bleu" in metrics:
        try:
            import sacrebleu
        except ImportError as error:
            raise RuntimeError("BLEU requires sacrebleu; install requirements.txt") from error
        results["bleu"] = float(
            sacrebleu.corpus_bleu(predictions, [references]).score
        )
    if "bertscore" in metrics:
        try:
            from bert_score import score as bert_score
        except ImportError as error:
            raise RuntimeError(
                "BERTScore requires bert-score; install requirements.txt"
            ) from error
        _, _, f1 = bert_score(
            predictions,
            references,
            lang="en",
            model_type="roberta-large",
            device=device,
            verbose=False,
        )
        results["bertscore_f1"] = float(f1.mean().item() * 100)
    if "lens" in metrics:
        try:
            from lens import LENS, download_model
        except ImportError as error:
            raise RuntimeError(
                "LENS requires lens-metric; install requirements-lens.txt"
            ) from error
        lens = LENS(download_model("davidheineman/lens"), rescale=True)
        devices = [0] if device.startswith("cuda") else None
        scores = lens.score(
            sources, predictions, [[reference] for reference in references],
            batch_size=8, devices=devices
        )
        results["lens"] = float(statistics.mean(scores))
    return results


def write_predictions(
    output_path: Path,
    records: list[dict[str, str]],
    predictions: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(
        [key for record in records for key in record] + ["prediction"]
    ))
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record, prediction in zip(records, predictions):
            writer.writerow({**record, "prediction": prediction})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_transformers():
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "Training and model inference require dependencies in requirements.txt"
        ) from error
    return (
        torch,
        Dataset,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )


def tokenize_records(
    records: list[dict[str, str]],
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
    include_targets: bool,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        [record["source"] for record in records],
        max_length=max_source_length,
        truncation=True,
    )
    if include_targets:
        targets = tokenizer(
            text_target=[record["reference"] for record in records],
            max_length=max_target_length,
            truncation=True,
        )
        encoded["labels"] = targets["input_ids"]
    return [
        {key: encoded[key][index] for key in encoded}
        for index in range(len(records))
    ]


def train(args: argparse.Namespace) -> None:
    (
        torch,
        Dataset,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    ) = load_transformers()
    set_seed(args.seed)
    train_path = args.train_file or default_data_file(args.task, "train")
    validation_path = args.validation_file or default_data_file(args.task, "val")
    train_records = read_records(train_path, args.task)
    validation_records = read_records(validation_path, args.task)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.revision
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, revision=args.revision
    )

    class EncodedDataset(Dataset):
        def __init__(self, records):
            self.items = tokenize_records(
                records, tokenizer, args.max_source_length, args.max_target_length, True
            )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        fp16=args.fp16,
        report_to="none",
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=EncodedDataset(train_records),
        eval_dataset=EncodedDataset(validation_records),
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    config = {
        "task": args.task,
        "model_name": args.model_name,
        "model_revision": (
            args.revision
            or getattr(model.config, "_commit_hash", None)
            or getattr(tokenizer, "_commit_hash", None)
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "logging_steps": args.logging_steps,
        "fp16": args.fp16,
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "validation_file": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    records_path = args.data_file or default_data_file(args.task, args.split)
    records = read_records(records_path, args.task)
    sources = [record["source"] for record in records]
    references = [record["reference"] for record in records]

    if args.model_path.lower() == "copy":
        predictions = sources.copy()
        device = "cpu"
    else:
        (
            torch,
            _,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            _,
            _,
            _,
        ) = load_transformers()
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        predictions = []
        for start in range(0, len(sources), args.batch_size):
            batch = sources[start : start + args.batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
            ).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_target_length,
                    num_beams=args.num_beams,
                    do_sample=False,
                )
            predictions.extend(
                tokenizer.batch_decode(output, skip_special_tokens=True)
            )

    metrics = calculate_metrics(
        sources, predictions, references, args.metrics, device
    )
    prediction_path = Path(
        args.predictions or f"results/{args.task}_{args.split}_predictions.csv"
    )
    metrics_path = Path(
        args.metrics_output or f"results/{args.task}_{args.split}_metrics.json"
    )
    write_predictions(prediction_path, records, predictions)
    metrics_payload = {
        "task": args.task,
        "split": args.split,
        "model": args.model_path,
        "data_file": str(records_path),
        "data_sha256": sha256_file(records_path),
        "example_count": len(records),
        "metrics": metrics,
        "metric_versions": {
            name: importlib.metadata.version(
                {"bleu": "sacrebleu", "bertscore": "bert-score", "lens": "lens-metric"}[name]
            )
            for name in args.metrics
            if name != "sari"
        },
        "sari_definition": (
            "local implementation; lowercase whitespace tokens; "
            "sentence-level mean; one reference; 1-4 gram orders"
            if "sari" in args.metrics
            else None
        ),
        "evaluation_settings": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "num_beams": args.num_beams,
        },
        "python_version": platform.python_version(),
        "predictions_file": str(prediction_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics_payload, indent=2))


def add_common_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=("sentence", "document"), required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune a BART baseline")
    add_common_task_arguments(train_parser)
    train_parser.add_argument("--model-name", default=DEFAULT_MODEL)
    train_parser.add_argument("--revision", help="Optional immutable Hugging Face revision")
    train_parser.add_argument("--train-file", type=Path)
    train_parser.add_argument("--validation-file", type=Path)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=float, default=3.0)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--max-source-length", type=int)
    train_parser.add_argument("--max-target-length", type=int)
    train_parser.add_argument("--logging-steps", type=int, default=100)
    train_parser.add_argument("--fp16", action="store_true")
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser(
        "evaluate", help="Generate outputs and calculate automatic metrics"
    )
    add_common_task_arguments(eval_parser)
    eval_parser.add_argument("--split", choices=("val", "test"), required=True)
    eval_parser.add_argument(
        "--model-path", default="copy", help="Model checkpoint directory or 'copy'"
    )
    eval_parser.add_argument("--data-file", type=Path)
    eval_parser.add_argument(
        "--metrics", nargs="+",
        choices=("sari", "bleu", "bertscore", "lens"),
        default=("sari", "bleu", "bertscore"),
    )
    eval_parser.add_argument("--predictions", type=Path)
    eval_parser.add_argument("--metrics-output", type=Path)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--max-source-length", type=int)
    eval_parser.add_argument("--max-target-length", type=int)
    eval_parser.add_argument("--num-beams", type=int, default=4)
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        args.max_source_length = args.max_source_length or (
            512 if args.task == "sentence" else 1024
        )
        args.max_target_length = args.max_target_length or (
            128 if args.task == "sentence" else 256
        )
    elif args.command == "evaluate":
        args.max_source_length = args.max_source_length or (
            512 if args.task == "sentence" else 1024
        )
        args.max_target_length = args.max_target_length or (
            128 if args.task == "sentence" else 256
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()

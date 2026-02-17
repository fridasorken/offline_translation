from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "finetune_config.json"


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


def _load_config() -> tuple[dict, dict, dict]:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return raw["paths"], raw["dataset"], raw["training"]


def _copy_best_checkpoint(trainer: Seq2SeqTrainer, best_model_dir: Path) -> str | None:
    best_checkpoint = trainer.state.best_model_checkpoint
    if not best_checkpoint:
        return None

    source = Path(best_checkpoint)
    if not source.exists():
        return None

    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    shutil.copytree(source, best_model_dir)
    return str(source)


def _find_last_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        (
            p
            for p in output_dir.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-") and p.name.split("-")[-1].isdigit()
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def main() -> None:
    path_cfg, dataset_cfg, training_cfg = _load_config()

    train_jsonl = _resolve_path(path_cfg["hf_train_jsonl"])
    eval_jsonl = _resolve_path(path_cfg["hf_eval_jsonl"])
    output_dir = _resolve_path(path_cfg["training_output_dir"])
    best_model_dir = _resolve_path(path_cfg["best_model_dir"])
    metrics_path = _resolve_path(path_cfg["training_metrics_json"])

    source_lang = training_cfg["source_lang"]
    target_lang = training_cfg["target_lang"]
    model_ref = training_cfg["model_ref"]
    local_files_only = bool(training_cfg.get("local_files_only", False))

    max_source_length = int(dataset_cfg["max_source_length"])
    max_target_length = int(dataset_cfg["max_target_length"])

    data_files = {"train": str(train_jsonl)}
    has_eval = eval_jsonl.exists() and eval_jsonl.stat().st_size > 0
    if has_eval:
        data_files["eval"] = str(eval_jsonl)

    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ref, local_files_only=local_files_only)
    # Some saved checkpoints can be loaded in fp16; AMP expects master weights in fp32.
    # Casting here avoids GradScaler "Attempting to unscale FP16 gradients" failures.
    train_dtype_set = {p.dtype for p in model.parameters() if p.requires_grad}
    if torch.float16 in train_dtype_set:
        model = model.float()
        print("cast_model_to_fp32_for_training=true")

    def preprocess(examples: dict) -> dict:
        source_texts = [f">>{target_lang}<< {text}" for text in examples["source"]]
        model_inputs = tokenizer(
            source_texts,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["reference"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_preds: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        vocab_size = len(tokenizer)

        predictions = predictions.astype(np.int64, copy=False)
        labels = labels.astype(np.int64, copy=False)
        labels = np.where(labels != -100, labels, pad_id)

        pred_invalid = (predictions < 0) | (predictions >= vocab_size)
        label_invalid = (labels < 0) | (labels >= vocab_size)
        if pred_invalid.any():
            predictions = np.where(pred_invalid, pad_id, predictions)
        if label_invalid.any():
            labels = np.where(label_invalid, pad_id, labels)

        pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = sacrebleu.corpus_bleu(pred_text, [label_text]).score
        chrf = sacrebleu.corpus_chrf(pred_text, [label_text]).score
        ter = sacrebleu.corpus_ter(pred_text, [label_text]).score
        return {"bleu": float(bleu), "chrf": float(chrf), "ter": float(ter)}

    evaluation_strategy = "steps" if has_eval else "no"
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
    train_rows = max(1, len(tokenized["train"]))
    per_device_bs = max(1, int(training_cfg["per_device_train_batch_size"]))
    grad_accum = max(1, int(training_cfg["gradient_accumulation_steps"]))
    epochs = max(1.0, float(training_cfg["num_train_epochs"]))
    steps_per_epoch = max(1, int(np.ceil(train_rows / (per_device_bs * grad_accum))))
    total_steps_est = int(np.ceil(steps_per_epoch * epochs))
    warmup_steps_est = int(max(0.0, warmup_ratio) * total_steps_est)

    requested_args: dict[str, object] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "learning_rate": float(training_cfg["learning_rate"]),
        "num_train_epochs": float(training_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps_est,
        "weight_decay": float(training_cfg["weight_decay"]),
        "logging_steps": int(training_cfg["logging_steps"]),
        "save_steps": int(training_cfg["save_steps"]),
        "eval_steps": int(training_cfg["eval_steps"]),
        "save_total_limit": int(training_cfg["save_total_limit"]),
        "predict_with_generate": True,
        "generation_num_beams": int(training_cfg["num_beams_eval"]),
        "fp16": bool(training_cfg.get("fp16", False)),
        "bf16": bool(training_cfg.get("bf16", False)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", False)),
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 0)),
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "bleu" if has_eval else None,
        "greater_is_better": True if has_eval else None,
        "report_to": [],
    }

    supported_args = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "warmup_steps" in supported_args:
        # Prefer warmup_steps to avoid warmup_ratio deprecation warnings in newer transformers.
        requested_args.pop("warmup_ratio", None)
    elif "warmup_ratio" not in supported_args:
        requested_args.pop("warmup_ratio", None)
        requested_args.pop("warmup_steps", None)

    if "evaluation_strategy" in supported_args:
        requested_args["evaluation_strategy"] = evaluation_strategy
    elif "eval_strategy" in supported_args:
        requested_args["eval_strategy"] = evaluation_strategy

    training_kwargs = {
        key: value
        for key, value in requested_args.items()
        if key in supported_args and value is not None
    }
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs: dict[str, object] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized.get("eval"),
        "data_collator": data_collator,
        "compute_metrics": compute_metrics if has_eval else None,
    }
    supported_trainer_args = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in supported_trainer_args:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in supported_trainer_args:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    resume_from_checkpoint = _find_last_checkpoint(output_dir)
    if resume_from_checkpoint:
        print(f"resume_from_checkpoint={resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_metrics = dict(train_result.metrics)

    eval_metrics: dict[str, float] = {}
    if has_eval:
        eval_metrics = dict(trainer.evaluate())

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    best_checkpoint = _copy_best_checkpoint(trainer, best_model_dir) if has_eval else None
    if not has_eval:
        if best_model_dir.exists():
            shutil.rmtree(best_model_dir)
        shutil.copytree(output_dir, best_model_dir)
        best_checkpoint = str(output_dir)

    metrics_payload = {
        "config_path": str(CONFIG_PATH),
        "model_ref": model_ref,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "train_rows": len(tokenized["train"]),
        "eval_rows": len(tokenized["eval"]) if has_eval else 0,
        "output_dir": str(output_dir),
        "best_model_dir": str(best_model_dir),
        "best_checkpoint": best_checkpoint,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"config_path={CONFIG_PATH}")
    print(f"model_ref={model_ref}")
    print(f"train_rows={len(tokenized['train'])}")
    print(f"eval_rows={len(tokenized['eval']) if has_eval else 0}")
    print(f"output_dir={output_dir}")
    print(f"best_model_dir={best_model_dir}")
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()

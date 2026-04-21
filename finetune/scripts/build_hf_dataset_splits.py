from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "finetune_config.json"
DEFAULT_EVAL_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.0
DEFAULT_RANDOM_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/eval/test splits from a JSONL dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to finetune config JSON (default: finetune/config/finetune_config.json)",
    )
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


def _load_config(config_path: Path) -> tuple[dict, dict, dict]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return raw["paths"], raw["dataset"], raw["training"]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_rows_by_conversation(rows: list[dict]) -> dict[str, list[dict]]:
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        conversation_id = str(row.get("conversation_id", "conv_missing"))
        grouped_rows[conversation_id].append(row)
    return grouped_rows


def _select_split_conversations(
    conversation_ids: list[str],
    grouped_rows: dict[str, list[dict]],
    target_eval_rows: int,
    target_test_rows: int,
) -> tuple[set[str], set[str]]:
    eval_conversations: set[str] = set()
    test_conversations: set[str] = set()
    eval_row_count = 0
    test_row_count = 0

    for conversation_id in conversation_ids:
        conversation_size = len(grouped_rows[conversation_id])
        if test_row_count < target_test_rows:
            test_conversations.add(conversation_id)
            test_row_count += conversation_size
            continue
        if eval_row_count < target_eval_rows:
            eval_conversations.add(conversation_id)
            eval_row_count += conversation_size

    return eval_conversations, test_conversations


def _build_output_row(row: dict, model_ref: str) -> dict:
    return {
        "item_id": row["item_id"],
        "conversation_id": row["conversation_id"],
        "turn_index": row["turn_index"],
        "source": row["source"],
        "reference": row["reference"],
        "domain": row["domain"],
        "risk_level": row["risk_level"],
        "model_ref": model_ref,
    }


def main() -> None:
    args = _parse_args()
    config_path = args.config
    path_cfg, dataset_cfg, training_cfg = _load_config(config_path)
    train_ready_path = _resolve_path(path_cfg["train_ready_jsonl"])
    train_path = _resolve_path(path_cfg["hf_train_jsonl"])
    eval_path = _resolve_path(path_cfg["hf_eval_jsonl"])
    test_path = _resolve_path(path_cfg["hf_test_jsonl"])
    report_path = _resolve_path(path_cfg["split_report_json"])

    eval_ratio = float(dataset_cfg.get("eval_ratio", DEFAULT_EVAL_RATIO))
    test_ratio = float(dataset_cfg.get("test_ratio", DEFAULT_TEST_RATIO))
    seed = int(dataset_cfg.get("random_seed", DEFAULT_RANDOM_SEED))

    rows = _read_jsonl(train_ready_path)
    if not rows:
        raise RuntimeError(f"Empty train-ready dataset: {train_ready_path}")

    grouped_rows = _group_rows_by_conversation(rows)
    conversation_ids = list(grouped_rows)
    random.Random(seed).shuffle(conversation_ids)

    total_rows = len(rows)
    target_eval_rows = int(total_rows * eval_ratio)
    target_test_rows = int(total_rows * test_ratio)
    eval_conversations, test_conversations = _select_split_conversations(
        conversation_ids=conversation_ids,
        grouped_rows=grouped_rows,
        target_eval_rows=target_eval_rows,
        target_test_rows=target_test_rows,
    )

    train_rows: list[dict] = []
    eval_rows_out: list[dict] = []
    test_rows_out: list[dict] = []

    for conversation_id in conversation_ids:
        for row in grouped_rows[conversation_id]:
            item = _build_output_row(row, training_cfg["model_ref"])
            if conversation_id in test_conversations:
                test_rows_out.append(item)
            elif conversation_id in eval_conversations:
                eval_rows_out.append(item)
            else:
                train_rows.append(item)

    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows_out)
    _write_jsonl(test_path, test_rows_out)

    report = {
        "config_path": str(config_path),
        "input_train_ready_path": str(train_ready_path),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "test_path": str(test_path),
        "seed": seed,
        "total_rows": total_rows,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows_out),
        "test_rows": len(test_rows_out),
        "eval_ratio_effective": round(len(eval_rows_out) / total_rows, 6),
        "test_ratio_effective": round(len(test_rows_out) / total_rows, 6),
        "train_ratio_effective": round(len(train_rows) / total_rows, 6),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"config_path={config_path}")
    print(f"train_ready_path={train_ready_path}")
    print(f"train_rows={len(train_rows)}")
    print(f"eval_rows={len(eval_rows_out)}")
    print(f"test_rows={len(test_rows_out)}")
    print(f"split_report={report_path}")


if __name__ == "__main__":
    main()

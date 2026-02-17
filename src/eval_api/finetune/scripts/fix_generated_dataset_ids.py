from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "finetune_config.json"

ITEM_ID_PREFIX = "gen_en_nob"
ITEM_ID_WIDTH = 6
CONVERSATION_ID_WIDTH = 4


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


def _load_paths() -> tuple[Path, Path, Path, int]:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    path_cfg = raw["paths"]
    id_cfg = raw.get("ids", {})

    input_dataset_path = _resolve_path(path_cfg["raw_generated_jsonl"])
    output_dataset_path = _resolve_path(path_cfg["fixed_ids_jsonl"])
    malformed_lines_path = _resolve_path(path_cfg["malformed_lines_log"])
    batch_id = int(id_cfg.get("batch_id", 1))
    return input_dataset_path, output_dataset_path, malformed_lines_path, batch_id


def main() -> None:
    input_dataset_path, output_dataset_path, malformed_lines_path, batch_id = _load_paths()

    records: list[dict[str, Any]] = []
    malformed_lines: list[tuple[int, str, str]] = []

    with input_dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                malformed_lines.append((line_number, str(exc), stripped))
                continue
            if not isinstance(obj, dict):
                malformed_lines.append((line_number, "line is not a JSON object", stripped))
                continue
            obj["_line_number"] = line_number
            records.append(obj)

    # Preserve conversation membership but normalize IDs and turn indexes.
    grouped: dict[str, list[dict[str, Any]]] = {}
    conversation_order: dict[str, int] = {}
    for row in records:
        raw_conversation_id = str(row.get("conversation_id") or "").strip()
        if not raw_conversation_id:
            raw_conversation_id = f"__missing_conv_{row['_line_number']}"
        if raw_conversation_id not in conversation_order:
            conversation_order[raw_conversation_id] = len(conversation_order) + 1
        grouped.setdefault(raw_conversation_id, []).append(row)

    conversation_id_map: dict[str, str] = {}
    for old_id, order in sorted(conversation_order.items(), key=lambda item: item[1]):
        new_conversation_id = f"conv_{order:0{CONVERSATION_ID_WIDTH}d}"
        conversation_id_map[old_id] = new_conversation_id

        rows = grouped[old_id]
        rows.sort(key=lambda row: (_safe_int(row.get("turn_index"), 10**9), row["_line_number"]))
        for new_turn_index, row in enumerate(rows, start=1):
            row["conversation_id"] = new_conversation_id
            row["turn_index"] = new_turn_index

    records.sort(key=lambda row: row["_line_number"])
    for new_item_index, row in enumerate(records):
        row["item_id"] = f"{ITEM_ID_PREFIX}_{batch_id}_{new_item_index:0{ITEM_ID_WIDTH}d}"
        row.pop("_line_number", None)

    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with output_dataset_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    malformed_lines_path.parent.mkdir(parents=True, exist_ok=True)
    with malformed_lines_path.open("w", encoding="utf-8") as handle:
        for line_number, reason, raw in malformed_lines:
            handle.write(f"line={line_number}\treason={reason}\traw={raw}\n")

    print(f"config_path={CONFIG_PATH}")
    print(f"input_path={input_dataset_path}")
    print(f"output_path={output_dataset_path}")
    print(f"malformed_log_path={malformed_lines_path}")
    print(f"valid_rows={len(records)}")
    print(f"malformed_rows={len(malformed_lines)}")
    print(f"conversations={len(conversation_order)}")


if __name__ == "__main__":
    main()

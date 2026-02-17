from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "finetune_config.json"

REQUIRED_FIELDS = (
    "item_id",
    "source",
    "reference",
    "conversation_id",
    "turn_index",
    "domain",
    "risk_level",
    "term_check_regex",
    "auto_filter",
)
ALLOWED_DOMAINS = {"ground_ops", "medical", "logistics", "artillery", "comms", "isr", "extraction", "roe"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
NOISY_PUNCTUATION = set(";:[]{}|")
REPEATED_TOKEN_RUN_THRESHOLD = 4
OVER_RUN_THRESHOLD = 3

NORWEGIAN_MARKERS = set("åøæÅØÆ")
NORWEGIAN_WORDS = {
    "og",
    "ikke",
    "til",
    "ved",
    "for",
    "med",
    "som",
    "på",
    "fra",
    "skal",
    "klar",
    "bekreft",
    "fienden",
    "over",
    "posisjonen",
    "sikre",
    "rapport",
    "anmodning",
    "mål",
    "styrker",
    "patrulje",
    "vi",
    "dere",
    "er",
}
ENGLISH_WORDS = {
    "the",
    "and",
    "is",
    "are",
    "to",
    "at",
    "with",
    "from",
    "hold",
    "fire",
    "report",
    "enemy",
    "grid",
    "confirm",
    "position",
    "request",
    "support",
    "ready",
    "move",
    "do",
    "not",
    "we",
    "you",
}


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


def _load_paths() -> tuple[Path, Path, Path]:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    path_cfg = raw["paths"]
    input_path = _resolve_path(path_cfg["fixed_ids_jsonl"])
    output_path = _resolve_path(path_cfg["train_ready_jsonl"])
    report_path = _resolve_path(path_cfg["train_ready_report_json"])
    return input_path, output_path, report_path


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _tokenize_words(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"[A-Za-zÅØÆåøæ']+", text)]


def _has_repeated_token_runs(text: str) -> bool:
    tokens = _tokenize_words(text)
    if not tokens:
        return False

    run_token = tokens[0]
    run_count = 1
    for token in tokens[1:]:
        if token == run_token:
            run_count += 1
        else:
            run_token = token
            run_count = 1

        if run_count >= REPEATED_TOKEN_RUN_THRESHOLD:
            return True
        if run_token == "over" and run_count >= OVER_RUN_THRESHOLD:
            return True
    return False


def _is_likely_non_english_source(text: str) -> bool:
    tokens = _tokenize_words(text)
    if not tokens:
        return False
    norwegian_score = sum(1 for token in tokens if token in NORWEGIAN_WORDS)
    if any(char in NORWEGIAN_MARKERS for char in text):
        norwegian_score += 3
    english_score = sum(1 for token in tokens if token in ENGLISH_WORDS)
    return norwegian_score >= 3 and english_score <= 1


def _row_quality(row: dict[str, Any]) -> tuple[int, int, int, int]:
    auto_filter = row["auto_filter"]
    return (
        int(auto_filter["adequacy_1to5"]),
        int(auto_filter["fluency_1to5"]),
        int(auto_filter["terminology_1to5"]),
        int(auto_filter["risk_criticality_1to5"]),
    )


def _validate_row(row: Any) -> tuple[bool, str]:
    if not isinstance(row, dict):
        return False, "not_json_object"

    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        return False, f"missing_fields:{','.join(missing)}"

    if row["domain"] not in ALLOWED_DOMAINS:
        return False, "invalid_domain"

    if row["risk_level"] not in ALLOWED_RISK_LEVELS:
        return False, "invalid_risk_level"

    if not isinstance(row["turn_index"], int) or row["turn_index"] < 1:
        return False, "invalid_turn_index"

    source = str(row["source"]).strip()
    reference = str(row["reference"]).strip()
    if not source:
        return False, "empty_source"
    if not reference:
        return False, "empty_reference"

    auto_filter = row["auto_filter"]
    if not isinstance(auto_filter, dict):
        return False, "invalid_auto_filter"

    if auto_filter.get("keep_for_finetune") is not True:
        return False, "not_kept_by_auto_filter"

    adequacy = auto_filter.get("adequacy_1to5")
    fluency = auto_filter.get("fluency_1to5")
    terminology = auto_filter.get("terminology_1to5")
    criticality = auto_filter.get("risk_criticality_1to5")
    scores = [adequacy, fluency, terminology, criticality]
    if any(not isinstance(score, int) for score in scores):
        return False, "invalid_auto_filter_scores"
    if any(score < 1 or score > 5 for score in scores):
        return False, "invalid_auto_filter_scores"
    if adequacy < 5 or fluency < 4 or terminology < 4:
        return False, "below_quality_threshold"

    reject_reasons = auto_filter.get("reject_reasons")
    if isinstance(reject_reasons, list) and reject_reasons:
        return False, "has_reject_reasons"

    normalized_source = _normalize_text(source)
    normalized_reference = _normalize_text(reference)
    if normalized_source == normalized_reference:
        return False, "source_equals_reference"

    if any(char in source for char in NOISY_PUNCTUATION):
        return False, "discouraged_punctuation_in_source"
    if any(char in reference for char in NOISY_PUNCTUATION):
        return False, "discouraged_punctuation_in_reference"
    if _has_repeated_token_runs(source):
        return False, "repeated_token_run_in_source"
    if _has_repeated_token_runs(reference):
        return False, "repeated_token_run_in_reference"

    term_check_regex = row.get("term_check_regex", "")
    if isinstance(term_check_regex, str) and term_check_regex:
        try:
            re.compile(term_check_regex)
        except re.error:
            return False, "invalid_term_check_regex"

    if _is_likely_non_english_source(source):
        return False, "likely_non_english_source"

    return True, "ok"


def main() -> None:
    input_path, output_path, report_path = _load_paths()

    removal_reasons: Counter[str] = Counter()
    removal_examples: dict[str, list[str]] = {}

    raw_rows = 0
    parsed_rows = 0
    candidates: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_rows += 1
            stripped = line.strip()
            if not stripped:
                removal_reasons["empty_line"] += 1
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                reason = "json_decode_error"
                removal_reasons[reason] += 1
                removal_examples.setdefault(reason, [])
                if len(removal_examples[reason]) < 10:
                    removal_examples[reason].append(f"line:{line_number}")
                continue

            parsed_rows += 1
            ok, reason = _validate_row(row)
            if not ok:
                removal_reasons[reason] += 1
                removal_examples.setdefault(reason, [])
                if len(removal_examples[reason]) < 10:
                    removal_examples[reason].append(str(row.get("item_id", f"line:{line_number}")))
                continue

            row["_quality"] = _row_quality(row)
            row["_source_norm"] = _normalize_text(str(row["source"]))
            row["_pair_norm"] = (
                row["_source_norm"],
                _normalize_text(str(row["reference"])),
            )
            candidates.append(row)

    # First deduplicate exact source+reference pairs.
    deduped_by_pair: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in candidates:
        key = row["_pair_norm"]
        if key in seen_pairs:
            removal_reasons["duplicate_source_reference_pair"] += 1
            continue
        seen_pairs.add(key)
        deduped_by_pair.append(row)

    # Then deduplicate by source text, keeping the highest quality version.
    best_by_source: dict[str, dict[str, Any]] = {}
    for row in deduped_by_pair:
        source_key = row["_source_norm"]
        previous = best_by_source.get(source_key)
        if previous is None or row["_quality"] > previous["_quality"]:
            if previous is not None:
                removal_reasons["duplicate_source_replaced_by_higher_quality"] += 1
            best_by_source[source_key] = row
        else:
            removal_reasons["duplicate_source_dropped"] += 1

    final_rows = list(best_by_source.values())
    final_rows.sort(key=lambda item: str(item["item_id"]))

    for row in final_rows:
        row.pop("_quality", None)
        row.pop("_source_norm", None)
        row.pop("_pair_norm", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    domain_distribution = Counter(row["domain"] for row in final_rows)
    risk_distribution = Counter(row["risk_level"] for row in final_rows)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "raw_rows": raw_rows,
        "parsed_rows": parsed_rows,
        "kept_rows": len(final_rows),
        "drop_rows_total": raw_rows - len(final_rows),
        "keep_rate": round(len(final_rows) / raw_rows, 6) if raw_rows else 0.0,
        "removal_reasons": dict(removal_reasons),
        "removal_examples": removal_examples,
        "domain_distribution": dict(domain_distribution),
        "risk_distribution": dict(risk_distribution),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"config_path={CONFIG_PATH}")
    print(f"input_path={input_path}")
    print(f"output_path={output_path}")
    print(f"report_path={report_path}")
    print(f"raw_rows={raw_rows}")
    print(f"kept_rows={len(final_rows)}")
    print(f"drop_rows={raw_rows - len(final_rows)}")


if __name__ == "__main__":
    main()

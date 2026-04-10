from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap source/reference in a JSONL dataset")
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSONL")
    return parser.parse_args()


def reverse_jsonl(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["source"], row["reference"] = row["reference"], row["source"]
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = _parse_args()
    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    count = reverse_jsonl(input_path, output_path)
    print(f"Reversed {count} rows: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()

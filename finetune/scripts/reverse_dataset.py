from __future__ import annotations

import argparse
import json
from pathlib import Path


def reverse_jsonl(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["source"], row["reference"] = row["reference"], row["source"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap source/reference in a JSONL dataset")
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    count = reverse_jsonl(input_path, output_path)
    print(f"Reversed {count} rows: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()

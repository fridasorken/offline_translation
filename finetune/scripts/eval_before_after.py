from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu  # type: ignore[import-untyped]
from comet import download_model, load_from_checkpoint  # type: ignore[import-untyped]
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


def load_config(config_path: Path) -> tuple[dict, dict, dict]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return raw["paths"], raw["dataset"], raw["training"]


def read_jsonl(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if len(rows) >= n:
                break
    return rows


def translate_batch(model, tokenizer, texts: list[str], use_target_prefix: bool,
                    target_lang: str, max_length: int = 192) -> list[str]:
    """Translate a list of texts using a HuggingFace seq2seq model."""
    if use_target_prefix:
        texts = [f">>{target_lang}<< {t}" for t in texts]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_length)
    outputs = model.generate(**inputs, num_beams=4, max_new_tokens=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def score(sources: list[str], hypotheses: list[str], references: list[str],
          comet_model) -> dict[str, float]:
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

    comet_data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    comet_output = comet_model.predict(comet_data, batch_size=8, gpus=0, num_workers=1)
    comet_score = comet_output.system_score

    return {"bleu": round(bleu, 2), "chrf": round(chrf, 2), "comet": round(comet_score, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50, help="Number of test samples")
    args = parser.parse_args()

    path_cfg, dataset_cfg, training_cfg = load_config(args.config)

    test_path = _resolve_path(path_cfg["hf_test_jsonl"])
    best_model_dir = _resolve_path(path_cfg["best_model_dir"])
    base_model_ref = training_cfg["model_ref"]
    source_lang = training_cfg["source_lang"]
    target_lang = training_cfg["target_lang"]
    use_target_prefix = bool(training_cfg.get("use_target_prefix", False))

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    samples = read_jsonl(test_path, args.n)
    sources = [s["source"] for s in samples]
    references = [s["reference"] for s in samples]

    print(f"Direction: {source_lang} -> {target_lang}")
    print(f"Base model: {base_model_ref}")
    print(f"Fine-tuned: {best_model_dir}")
    print(f"Test samples: {len(samples)}")
    print()

    # Load COMET
    print("Loading COMET model...")
    comet_checkpoint = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_checkpoint)

    # Base model
    print(f"Loading base model: {base_model_ref}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_ref)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_ref)
    base_model.eval()

    print("Translating with base model...")
    base_hyps = translate_batch(base_model, base_tokenizer, sources,
                                use_target_prefix, target_lang)
    base_scores = score(sources, base_hyps, references, comet_model)
    print(f"  BASE:      BLEU={base_scores['bleu']}  chrF={base_scores['chrf']}  COMET={base_scores['comet']}")

    # Free memory
    del base_model, base_tokenizer

    # Fine-tuned model
    print(f"Loading fine-tuned model: {best_model_dir}")
    ft_tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir))
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(str(best_model_dir))
    ft_model.eval()

    print("Translating with fine-tuned model...")
    ft_hyps = translate_batch(ft_model, ft_tokenizer, sources,
                              use_target_prefix, target_lang)
    ft_scores = score(sources, ft_hyps, references, comet_model)
    print(f"  FINETUNED: BLEU={ft_scores['bleu']}  chrF={ft_scores['chrf']}  COMET={ft_scores['comet']}")

    # Delta
    print()
    print("Delta (finetuned - base):")
    for metric in ("bleu", "chrf", "comet"):
        delta = ft_scores[metric] - base_scores[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric}: {sign}{round(delta, 4)}")


if __name__ == "__main__":
    main()

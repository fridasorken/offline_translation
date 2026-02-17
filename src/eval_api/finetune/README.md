# OPUS Fine-Tuning Workflow

This folder contains the full EN -> NOB fine-tuning pipeline for `Helsinki-NLP/opus-mt-tc-big-en-gmq`.

Everything is config-driven via:

- `src/eval_api/finetune/config/finetune_config.json`

No script arguments are required.

## Folder layout

- `data/`: cleaned and split datasets
- `results/`: QA/split/training reports
- `scripts/`: cleanup and training scripts
- `outputs/`: model checkpoints

## Run order

From `src/eval_api`:

```bash
uv sync
uv run python finetune/scripts/train_opus_tc_big.py
```

The repository includes ready-to-train split files (`opus_train.jsonl`, `opus_eval.jsonl`).
If you want to rebuild splits from raw generated data, run the preprocessing scripts first.

## Important config knobs

Edit `finetune_config.json` to control:

- source raw dataset path (`paths.raw_generated_jsonl`)
- cleaned datasets and reports paths (`paths.*`)
- split ratios (`dataset.eval_ratio`, `dataset.test_ratio`)
- max token lengths (`dataset.max_source_length`, `dataset.max_target_length`)
- model and training hyperparameters (`training.*`)

## Outputs

After training:

- main checkpoint output: `paths.training_output_dir`
- best model export: `paths.best_model_dir`
- training metrics: `paths.training_metrics_json`

## Git hygiene

This folder is set up so git tracks pipeline code/config/data/reports, but ignores heavy local artifacts:

- ignored: `model/`, `outputs/`, `__pycache__/`, `.DS_Store`
- tracked: `scripts/`, `config/`, `data/`, `results/`, this README

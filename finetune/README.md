# OPUS Fine-Tuning Project

This is a standalone fine-tuning project (separate from the runtime API in `src/eval_api`).

Everything is config-driven via:

- `finetune/config/finetune_config.json`

No script arguments are required.

## Folder layout

- `config/`: training and dataset config
- `data/`: source/train/eval/test datasets
- `scripts/`: dataset split + training scripts
- `outputs/`: training checkpoints (ignored)
- `model/`: local model artifacts (ignored)

## Run order

From repository root:

```bash
uv run python finetune/scripts/build_hf_dataset_splits.py
uv run python finetune/scripts/train_opus_tc_big.py
```

## Outputs

After training:

- main checkpoint output: `paths.training_output_dir`
- best model export: `paths.best_model_dir`
- training metrics: `paths.training_metrics_json`

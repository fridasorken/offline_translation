# Fine-Tune Guide

This folder is a small, config-driven fine-tuning workflow for OPUS-style seq2seq models.

The important constraint is simple:

- do not rely on CLI args for the normal workflow
- change values in `finetune/config/finetune_config.json` instead

This branch still has some legacy script arguments, but the intended workflow is still config-first.

## What is in this folder

The main training flow is:

1. `finetune/scripts/build_hf_dataset_splits.py`
2. `finetune/scripts/train_opus_tc_big.py`

This branch also includes helper scripts:

- `finetune/scripts/reverse_dataset.py`
- `finetune/scripts/eval_before_after.py`

## Config files

The default config is:

- `finetune/config/finetune_config.json`

This branch also has extra config variants:

- `finetune/config/finetune_config_de_en.json`
- `finetune/config/finetune_config_nob_en.json`
- `finetune/config/finetune_config_pt_en.json`

Preferred workflow:

1. Start from the config closest to what you want.
2. Copy its values into `finetune/config/finetune_config.json`.
3. Run the scripts without passing extra args.

The scripts on this branch technically support `--config`, but if you want a stable setup, it is cleaner to make the intended config the default file instead of depending on CLI flags.

## What the main scripts do

`build_hf_dataset_splits.py`:

- reads the train-ready JSONL file from `paths.train_ready_jsonl`
- groups rows by `conversation_id`
- creates train/eval/test splits
- writes normalized JSONL outputs and a split report

`train_opus_tc_big.py`:

- loads the split files
- tokenizes source/reference text
- fine-tunes the configured OPUS model
- saves checkpoints, a best-model copy, and metrics

## Path resolution

All relative paths in the config are resolved from the `finetune/` directory, not from your shell cwd.

## Expected input data

The split builder expects `paths.train_ready_jsonl` to contain rows with at least:

- `item_id`
- `conversation_id`
- `turn_index`
- `source`
- `reference`
- `domain`
- `risk_level`

The split builder keeps full conversations together by grouping on `conversation_id`.

The training script later reads:

- `paths.hf_train_jsonl`
- `paths.hf_eval_jsonl` if it exists and is non-empty

For training itself, the key fields are `source` and `reference`, but richer rows are fine.

## Current default config on this branch

Right now, `finetune/config/finetune_config.json` is set up for:

- model: `gsarti/opus-mt-tc-big-en-de`
- source language: `en`
- target language token/config value: `deu`
- train-ready input: `generated_en_de_military_finalized_v1_openrouter.jsonl`
- output dir: `finetune/outputs/opus_tc_big_en_de_ft_v1`

Other committed configs on this branch cover:

- `de -> en`
- `nob -> en`
- `pt -> en`

## How to run

Recommended from repository root:

```bash
python finetune/scripts/build_hf_dataset_splits.py
python finetune/scripts/train_opus_tc_big.py
```

If you prefer `uv`, the same flow is:

```bash
uv run python finetune/scripts/build_hf_dataset_splits.py
uv run python finetune/scripts/train_opus_tc_big.py
```

## Environment requirements

This branch does not currently include a dedicated `pyproject.toml` or `requirements.txt` for `finetune/`.

So before running the scripts, make sure your Python environment already has the required packages. Based on the script imports, that includes at least:

- `datasets`
- `numpy`
- `sacrebleu`
- `torch`
- `transformers`

For the evaluation helper, you also need:

- `comet`

In practice, Hugging Face training setups also often need `accelerate`.

## How To Generate Dataset

This branch does not currently include a committed dataset-generation script inside `finetune/`.

So in practice, there are three ways to create a new train-ready dataset. The way we did in the start was starting with option3, then option 1 when expanding to new languages:

1. Write a script that calls an LLM that calls an LLM through a provider such as OpenRouter and translate the current dataset in github to whatever language you need. This is what was done for german and portugese, translating the en->nob dataset.
2. Write or use a small script that calls an LLM through a provider such as OpenRouter and writes NDJSON rows to a file under `finetune/data/`.
3. Generate the dataset directly with a strong prompt, save the NDJSON response to a file under `finetune/data/`, and then point `paths.train_ready_jsonl` at that file.

There is also a practical reuse path already reflected in this branch:

- the reverse-direction datasets such as `opus_de_en_train.jsonl`, `opus_no_en_train.jsonl`, and `opus_pt_en_train.jsonl` were produced by translating or reversing an existing dataset into the direction needed for training

Whichever route you use, the final file should be JSONL/NDJSON and match the field layout expected by the split builder:

- `item_id`
- `conversation_id`
- `turn_index`
- `source`
- `reference`
- `domain`
- `risk_level`

The extra fields in the example prompt below are fine to keep. The split builder preserves the core training fields and writes a normalized train/eval/test version afterward.

### Suggested prompt for English -> Norwegian Bokmal generation

Use a prompt like this when generating targeted synthetic data:

```text
You are generating high-quality parallel data for English -> Norwegian Bokmal military-domain MT fine-tuning.

Task:
Generate EXACTLY 100 dataset rows.
Return ONLY NDJSON (one JSON object per line), no markdown, no explanations.

Hard requirements:
- Source language: English.
- Target language: Norwegian Bokmal.
- One source sentence and one target sentence per row.
- Preserve meaning exactly (no added/removed facts).
- Preserve all slot values exactly: coordinates, callsigns, times, frequencies/channels, unit numbers, bearings, distances, and quantities.
- Preserve negation and command polarity exactly.
- Prefer operationally correct military wording in Norwegian field radio style.
- Use UTF-8 Norwegian characters (å, ø, æ).
- Avoid duplicates and near-duplicates.

Schema constraints:
- domain must be one of: [ground_ops, medical, logistics, artillery, comms, isr, extraction, roe]
- risk_level must be one of: [low, medium, high]
- Do NOT use domain "cbrn" (put CBRN-related content under "medical" or "comms")

Anti-leakage constraints:
- Do NOT copy these exact English source stems:
  1) "Triage update: one urgent surgical and"
  2) "Jamming pulse ended, restore primary net on"
  3) "Satellite pass window opens at"
  4) "Littoral team reports probable limpet mines on pier"
  5) "Sensor-to-shooter relay: mark target at grid"
  6) "Push firmware image hash"
  7) "jammer sweep shows intermittent interference on"
  8) "Route Granite is under intermittent indirect fire, move convoy by bounds of"
- Cover the same operational meaning but with different wording and structure.

Strict target-language bans (reject if any appear in reference):
- "skyt ut CASEVAC"
- "støysendersveip"
- "laveffekt-skurmodus"
- "nedkjempingspuls"
- "satellittpasseringsvindu"
- "firmware-bildehasj"
- "lokkedue" (use standard military wording)
- "svaber" (use standard wording for sample/swab in Norwegian context)
- "termisk mating"
- "treghetsruta"
- "kø for pasient"

Automatic filtering (self-judge each row):
- adequacy_1to5 >= 5
- fluency_1to5 >= 4
- terminology_1to5 >= 4
- Reject if literal/awkward Norwegian wording.
- Reject if slot values changed or missing.
- Reject duplicates/near-duplicates.

Output schema per NDJSON line:
{
  "item_id": "gen_en_nob_targeted_v2_000001",
  "source": "English sentence",
  "reference": "Norwegian Bokmal translation",
  "conversation_id": "conv_0001",
  "turn_index": 1,
  "domain": "one_of:[ground_ops,medical,logistics,artillery,comms,isr,extraction,roe]",
  "risk_level": "one_of:[low,medium,high]",
  "term_check_regex": "regex for key disambiguated term or empty string",
  "auto_filter": {
    "keep_for_finetune": true,
    "adequacy_1to5": 5,
    "fluency_1to5": 5,
    "terminology_1to5": 5,
    "risk_criticality_1to5": 3,
    "reject_reasons": []
  }
}
Batch focus:
- net degradation and restoration, fallback channels, timing windows, traffic prioritization.

Required phenomena:
- primary/alternate net transitions
- auth and key-rotation steps
- satellite window prioritization of casualty/fire traffic
- jam/interference recovery

Style focus:
- standard Norwegian military comms wording
- avoid awkward noun compounds and literal loan translations

Distribution:
- domains: comms 70%, isr 15%, logistics 15%
- risk: high >= 60%
- ASR-like rows: >= 20%

100 examples to gen_en_nob_targeted_v2_000300
```

### After generation

Once you have the NDJSON file:

1. Save it under `finetune/data/`.
2. Update `paths.train_ready_jsonl` to point to it.
3. If needed, also update `paths.raw_generated_jsonl` so the config still reflects the real source file.
4. Change the train/eval/test output filenames if you are creating a new dataset version.
5. Run the split builder.

## Step 1: Build dataset splits

Run:

```bash
python finetune/scripts/build_hf_dataset_splits.py
```

What it does:

- loads `paths.train_ready_jsonl`
- shuffles conversations with `dataset.random_seed`
- assigns whole conversations into train/eval/test buckets
- writes:
  - `paths.hf_train_jsonl`
  - `paths.hf_eval_jsonl`
  - `paths.hf_test_jsonl`
- writes a summary report to `paths.split_report_json`

Config knobs that usually matter here:

- `dataset.random_seed`
- `dataset.eval_ratio`
- `dataset.test_ratio`
- `paths.train_ready_jsonl`
- `paths.hf_train_jsonl`
- `paths.hf_eval_jsonl`
- `paths.hf_test_jsonl`

## Step 2: Train the model

Run:

```bash
python finetune/scripts/train_opus_tc_big.py
```

What it does:

- loads train data from `paths.hf_train_jsonl`
- loads eval data from `paths.hf_eval_jsonl` if present
- loads tokenizer and model from `training.model_ref`
- tokenizes `source` and `reference`
- optionally prefixes the source text with `training.target_prefix_template`
- trains using `Seq2SeqTrainer`
- saves the final model to `paths.training_output_dir`
- copies the best model to `paths.best_model_dir`
- writes training/eval metrics to `paths.training_metrics_json`

Important training config fields:

- `training.model_ref`
- `training.source_lang`
- `training.target_lang`
- `training.use_target_prefix`
- `training.target_prefix_template`
- `training.local_files_only`
- `training.num_train_epochs`
- `training.learning_rate`
- `training.per_device_train_batch_size`
- `training.gradient_accumulation_steps`
- `training.fp16`
- `training.bf16`
- `training.gradient_checkpointing`
- `dataset.max_source_length`
- `dataset.max_target_length`

## Resume behavior

The training script automatically looks for the latest checkpoint under `paths.training_output_dir`.

If it finds a folder like `checkpoint-300`, it resumes from the newest checkpoint automatically. You do not need to pass a resume flag.

## Eval behavior during and after training

During training, eval only runs if both of these are true:

- `paths.hf_eval_jsonl` exists and is non-empty
- `training.disable_eval_during_training` is `false`

After training, final eval only runs if:

- there is an eval dataset
- `training.skip_final_eval` is `false`

If eval is disabled, the script still saves the final model and copies that final output into `paths.best_model_dir`.

## Extra helper scripts on this branch

### `reverse_dataset.py`

This helper swaps `source` and `reference` in a JSONL file.

Use it when you already have a dataset in one direction and want to bootstrap the reverse direction before splitting and training.

Because this script currently requires explicit input/output args on this branch, it does not follow the same config-first pattern as the two main scripts.

### `eval_before_after.py`

This helper compares the base model and the fine-tuned model on the configured test set.

It scores with:

- BLEU
- chrF
- COMET

The important files it depends on are:

- `paths.hf_test_jsonl`
- `paths.best_model_dir`
- `training.model_ref`

This helper also still uses CLI args on this branch, so treat it as an evaluation utility rather than the main stable workflow.

## Outputs to check

After a normal run, the main outputs are:

- split report: `paths.split_report_json`
- training checkpoints/final export: `paths.training_output_dir`
- best model copy: `paths.best_model_dir`
- metrics JSON: `paths.training_metrics_json`

This branch already contains some example outputs under `finetune/results/`.

## Typical workflow when changing datasets

1. Put the new dataset JSONL somewhere under `finetune/data/`.
2. Update `paths.train_ready_jsonl`.
3. Update the train/eval/test output filenames in `paths`.
4. Adjust `dataset.eval_ratio` and `dataset.test_ratio` if needed.
5. Run the split builder.
6. Inspect `paths.split_report_json`.
7. Run training.

## Typical workflow when changing model or language pair

1. Start from the closest config file in `finetune/config/`.
2. Copy the desired values into `finetune/config/finetune_config.json`.
3. Update `training.model_ref`.
4. Update `training.source_lang` and `training.target_lang`.
5. Decide whether `training.use_target_prefix` should be enabled.
6. Change output paths so you do not overwrite an older experiment.
7. Rebuild splits if the dataset version also changed.
8. Run training.

## Common failure points

- Missing input file:
  `paths.train_ready_jsonl` does not exist.
- Empty dataset:
  the split builder raises `Empty train-ready dataset`.
- Missing dependencies:
  the environment on this branch is not declared in a committed dependency file.
- Wrong model prefix setup:
  `training.use_target_prefix` and `training.target_prefix_template` must match the model family you are training.
- Reusing old output paths:
  if you want separate experiments, change `training_output_dir`, `best_model_dir`, and `training_metrics_json` first.
- Forgetting to switch default config:
  this branch has multiple config files, but the main scripts default to `finetune/config/finetune_config.json`.

## Minimal run checklist

Before running:

- confirm `finetune/config/finetune_config.json` points to the right dataset
- confirm your output paths are versioned the way you want
- confirm your Python environment has the training dependencies installed

Run:

```bash
python finetune/scripts/build_hf_dataset_splits.py
python finetune/scripts/train_opus_tc_big.py
```

Check:

- split counts in `paths.split_report_json`
- saved checkpoints under `paths.training_output_dir`
- exported best model under `paths.best_model_dir`
- metrics in `paths.training_metrics_json`

# Task 2 Training Guide

This repository is the binary classification setup for the second shared task: for each `<question, candidate answer>` pair, predict whether the candidate is correct using the question text and the candidate graph.

I added a runnable training and inference pipeline in [task2_pipeline.py](/home/nabuki/src/TextGraphs17-shared-task/task2_pipeline.py) and fixed the evaluator in [evaluation/evaluate.py](/home/nabuki/src/TextGraphs17-shared-task/evaluation/evaluate.py).

## What I implemented

The pipeline follows the task definition in the repo and also incorporates the useful notebook ideas in a clean CLI form:

- It treats the problem as **binary classification per row**.
- It uses a **question-level train/validation split**, so all candidates from the same question stay together and there is no leakage.
- It supports three model families:
  `linearized_text`, `graph_stats`, and `cross_attention`.
- `linearized_text` converts each graph into a readable text sequence of triples and trains a standard transformer classifier.
- `graph_stats` combines transformer text features with handcrafted graph statistics inspired by the notebook.
- `cross_attention` combines transformer text features with graph node labels through a graph-aware cross-attention block inspired by the notebook.
- It searches for the **best validation threshold** for F1 and saves that threshold with the checkpoint.
- It writes ready-to-use TSV files for validation and test predictions.

## Model Types

The most important CLI option is:

```text
--model-type
```

Available values:

- `linearized_text`: baseline-style model. It linearizes the graph into text and feeds a single transformer input to the classifier.
- `graph_stats`: lightweight graph-aware model. It uses the question/answer text through the transformer and concatenates that with normalized handcrafted graph statistics.
- `cross_attention`: notebook-inspired graph-aware model. It uses the question/answer text through the transformer and fuses it with graph node label representations through multi-head cross-attention.

Practical guidance:

- Start with `graph_stats` if you want the fastest notebook-inspired comparison.
- Use `cross_attention` if you want the strongest graph-aware model in this repo.
- Use `linearized_text` if you want the simplest baseline and easiest debugging path.

## Files

- [task2_pipeline.py](/home/nabuki/src/TextGraphs17-shared-task/task2_pipeline.py): train / validate / test CLI
- [evaluation/evaluate.py](/home/nabuki/src/TextGraphs17-shared-task/evaluation/evaluate.py): fixed evaluator
- [pyproject.toml](/home/nabuki/src/TextGraphs17-shared-task/pyproject.toml): project metadata and Python dependencies

## Setup

Create the environment and install dependencies with `uv`:

```bash
uv sync
```

Then run everything through `uv run`:

```bash
uv run python3 task2_pipeline.py --help
```

The default encoder is:

```text
sentence-transformers/all-mpnet-base-v2
```

The default model family is:

```text
cross_attention
```

This is the notebook-inspired graph-aware option. It will download the encoder from Hugging Face the first time you run it.

For true multi-GPU training, launch the script with Hugging Face `accelerate`.

## Important Implementation Notes

- Validation F1 is not computed with a fixed threshold of `0.5`. The script searches for the best threshold on the validation split and saves it in `metadata.json`.
- The training split is by **question**, not by individual rows, to avoid leakage between candidate answers for the same question.
- `graph_stats` normalizes graph statistics using the training split only.
- `cross_attention` uses graph node labels as node representations and applies cross-attention from text `[CLS]` to those node representations.
- The current `cross_attention` implementation uses the transformer's input embedding layer plus masked pooling for node-label encoding, which is much cheaper than running the full transformer separately over every node label on every step.
- This is one reason `cross_attention` can train much faster than a naive notebook implementation.
- For multi-GPU runs launched with `accelerate`, each process gets its own shard of the data loader.

## Train

This command trains the default `cross_attention` model on `data/tsv/train.tsv`, creates a held-out validation split by question, saves the best checkpoint, and also writes validation predictions:

```bash
uv run python3 task2_pipeline.py train \
  --train-path data/tsv/train.tsv \
  --output-dir runs/task2_mpnet
```

For 2 GPUs, use:

```bash
uv run accelerate launch --multi_gpu task2_pipeline.py train \
  --train-path data/tsv/train.tsv \
  --output-dir runs/task2_mpnet
```

Useful optional arguments:

- `--model-type linearized_text`
- `--model-type graph_stats`
- `--model-type cross_attention`
- `--model-name roberta-base`
- `--epochs 5`
- `--batch-size 8`
- `--max-length 256`
- `--node-max-length 16`
- `--max-nodes 30`
- `--learning-rate 2e-5`
- `--val-ratio 0.1`
- `--freeze-embeddings`
- `--freeze-layers 5`
- `--seed 42`

When running with `accelerate`, `--batch-size` is the per-process batch size.

Less obvious but useful training flags:

- `--dropout`: default `0.2`
- `--grad-accum-steps`: gradient accumulation steps
- `--weight-decay`: default `0.01`
- `--warmup-ratio`: scheduler warmup ratio
- `--num-workers`: dataloader workers

Flags that matter by model type:

- `linearized_text`
  Uses `--max-length`
- `graph_stats`
  Uses `--max-length`
- `cross_attention`
  Uses `--max-length`, `--node-max-length`, and `--max-nodes`

Freezing controls:

- `--freeze-embeddings`: freezes the encoder embedding layer
- `--freeze-layers N`: freezes the first `N` transformer encoder layers

These are especially useful if the graph-aware head is overfitting or if you want behavior closer to the notebook experiments.

Examples:

```bash
uv run python3 task2_pipeline.py train \
  --model-type graph_stats \
  --train-path data/tsv/train.tsv \
  --output-dir runs/task2_graph_stats
```

```bash
uv run accelerate launch --multi_gpu task2_pipeline.py train \
  --model-type cross_attention \
  --train-path data/tsv/train.tsv \
  --output-dir runs/task2_cross_attention
```

```bash
uv run accelerate launch --multi_gpu task2_pipeline.py train \
  --model-type cross_attention \
  --train-path data/tsv/train.tsv \
  --output-dir runs/task2_cross_attention_tuned \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --dropout 0.2 \
  --max-nodes 30 \
  --node-max-length 16
```

Artifacts written to `runs/task2_mpnet/`:

- `best_model.pt`
- `metadata.json`
- `split.json`
- `tokenizer/`
- `val_predictions.tsv`

What is stored in `metadata.json`:

- `model_name`
- `model_type`
- `dropout`
- `max_length`
- `node_max_length`
- `max_nodes`
- `freeze_embeddings`
- `freeze_layers`
- `threshold`
- `seed`
- `val_ratio`
- validation metrics for the best checkpoint

## Validate

This reruns the saved model on the exact held-out validation split and prints the metrics:

```bash
uv run python3 task2_pipeline.py validate \
  --train-path data/tsv/train.tsv \
  --checkpoint-dir runs/task2_mpnet
```

If you want to explicitly score the saved validation file with the evaluator:

```bash
uv run python3 evaluation/evaluate.py \
  --predictions_path runs/task2_mpnet/val_predictions.tsv \
  --gold_labels_path data/tsv/train.tsv
```

Note:

- `evaluation/evaluate.py` aligns predictions by `sample_id`.
- `val_predictions.tsv` only contains the held-out validation rows, not the full train file.
- The reliable validation numbers are the ones printed by:
  `python3 task2_pipeline.py validate ...`
- `validate` reloads the saved model configuration from `metadata.json`, so you do not need to pass `--model-type` again.

## Test

This loads the saved checkpoint and writes a prediction file for `data/tsv/test.tsv`:

```bash
uv run python3 task2_pipeline.py test \
  --test-path data/tsv/test.tsv \
  --checkpoint-dir runs/task2_mpnet \
  --output-path runs/task2_mpnet/test_predictions.tsv
```

The produced file contains:

- `sample_id`
- `probability`
- `prediction`

If you need a submission file with only the required columns, keep:

- `sample_id`
- `prediction`

## Speed Notes

Training speed depends a lot on model type:

- `graph_stats` is usually the fastest.
- `linearized_text` is usually in the middle.
- `cross_attention` is graph-aware and more expensive than `graph_stats`, but still much faster than a design that would run the full transformer on every graph node label at every step.

If training suddenly becomes much faster than an older notebook run, that can be expected if:

- you switched to multi-GPU `accelerate`
- you changed to `graph_stats`
- you reduced `max_nodes` or `node_max_length`
- your older notebook was re-encoding graph labels much more expensively

If you compare models, compare them on:

- the same `--seed`
- the same split
- the same effective batch size
- the same number of epochs

## Recommended run order

```bash
uv run python3 task2_pipeline.py train --train-path data/tsv/train.tsv --output-dir runs/task2_mpnet
uv run python3 task2_pipeline.py validate --train-path data/tsv/train.tsv --checkpoint-dir runs/task2_mpnet
uv run python3 task2_pipeline.py test --test-path data/tsv/test.tsv --checkpoint-dir runs/task2_mpnet --output-path runs/task2_mpnet/test_predictions.tsv
```

## Notes

- The train set is highly imbalanced, so the training loss uses a positive-class weight automatically.
- Some questions have more than one correct candidate, so this is not implemented as single-choice ranking.
- The script does not require `pandas`, `numpy`, or `scikit-learn`.
- If GPU memory is tight, reduce `--batch-size` first.
- For multi-GPU runs, prefer `accelerate launch --multi_gpu ...` instead of raw `python3 ...`.
- `graph_stats` is the lightest notebook-inspired model and is a good first comparison point.
- `cross_attention` is the strongest notebook-inspired graph-aware model in this repo and usually needs more careful tuning of batch size, dropout, and learning rate.
- If validation F1 changes after moving from 1 GPU to multi-GPU, the first thing to check is effective batch size. With `accelerate`, `--batch-size` is per process.

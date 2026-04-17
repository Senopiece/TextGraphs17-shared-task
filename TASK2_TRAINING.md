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

Artifacts written to `runs/task2_mpnet/`:

- `best_model.pt`
- `metadata.json`
- `split.json`
- `tokenizer/`
- `val_predictions.tsv`

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

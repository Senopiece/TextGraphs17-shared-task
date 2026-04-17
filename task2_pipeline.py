from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run a transformer baseline for TextGraphs shared task 2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train on train.tsv and evaluate on a held-out validation split.")
    add_common_model_args(train_parser)
    train_parser.add_argument("--train-path", type=str, default="data/tsv/train.tsv")
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--val-ratio", type=float, default=0.1)
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=0)

    validate_parser = subparsers.add_parser("validate", help="Run the saved checkpoint on the held-out validation split.")
    validate_parser.add_argument("--train-path", type=str, default="data/tsv/train.tsv")
    validate_parser.add_argument("--checkpoint-dir", type=str, required=True)
    validate_parser.add_argument("--output-path", type=str, default="")
    validate_parser.add_argument("--batch-size", type=int, default=32)
    validate_parser.add_argument("--num-workers", type=int, default=0)

    test_parser = subparsers.add_parser("test", help="Run the saved checkpoint on the hidden-label test set.")
    test_parser.add_argument("--test-path", type=str, default="data/tsv/test.tsv")
    test_parser.add_argument("--checkpoint-dir", type=str, required=True)
    test_parser.add_argument("--output-path", type=str, required=True)
    test_parser.add_argument("--batch-size", type=int, default=32)
    test_parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Hugging Face encoder to fine-tune.",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_graph(graph_value: object) -> Dict[str, object]:
    if isinstance(graph_value, dict):
        return graph_value
    if not isinstance(graph_value, str):
        raise TypeError(f"Unsupported graph value type: {type(graph_value)!r}")
    return ast.literal_eval(graph_value)


def linearize_graph(graph_value: object) -> str:
    graph = parse_graph(graph_value)
    nodes = sorted(graph["nodes"], key=lambda item: item["id"])
    node_by_id = {node["id"]: node for node in nodes}

    outgoing: Dict[int, List[Dict[str, object]]] = {}
    for edge in graph["links"]:
        outgoing.setdefault(edge["source"], []).append(edge)

    triples: List[str] = []
    for node in nodes:
        source_label = decorate_node_label(node)
        for edge in outgoing.get(node["id"], []):
            target_label = decorate_node_label(node_by_id[edge["target"]])
            relation_label = str(edge["label"]).strip()
            triples.append(f"{source_label} -> {relation_label} -> {target_label}")

    if not triples:
        return "NO_GRAPH_EDGES"
    return " [GRAPH_SEP] ".join(triples)


def decorate_node_label(node: Dict[str, object]) -> str:
    node_type = str(node.get("type", "NODE"))
    label = str(node.get("label", "")).strip()
    if node_type == "QUESTIONS_ENTITY":
        return f"[QUESTION_ENTITY] {label}"
    if node_type == "ANSWER_CANDIDATE_ENTITY":
        return f"[ANSWER_ENTITY] {label}"
    return f"[INTERNAL_ENTITY] {label}"


def build_model_input(row: Dict[str, str]) -> str:
    parts = [
        f"Question: {row['question'].strip()}",
        f"Question entities: {row.get('questionEntity', '').strip()}",
        f"Candidate answer: {row['answerEntity'].strip()}",
        f"Graph: {linearize_graph(row['graph'])}",
    ]
    return " [SEP] ".join(parts)


def label_from_row(row: Dict[str, str]) -> int:
    return 1 if row["correct"] == "True" else 0


def split_rows_by_question(
    rows: Sequence[Dict[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, List[str]]]:
    questions = sorted({row["question"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(questions)

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in (0, 1), got {val_ratio}.")

    num_val_questions = max(1, int(round(len(questions) * val_ratio)))
    num_val_questions = min(num_val_questions, len(questions) - 1)

    val_questions = set(questions[:num_val_questions])
    train_questions = set(questions[num_val_questions:])

    train_rows = [row for row in rows if row["question"] in train_questions]
    val_rows = [row for row in rows if row["question"] in val_questions]

    split_payload = {
        "train_questions": sorted(train_questions),
        "val_questions": sorted(val_questions),
    }
    return train_rows, val_rows, split_payload


@dataclass
class Example:
    sample_id: str
    text: str
    label: int | None


class Task2Dataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], include_labels: bool) -> None:
        self.examples = [
            Example(
                sample_id=row["sample_id"],
                text=build_model_input(row),
                label=label_from_row(row) if include_labels else None,
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


class Collator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Example]) -> Dict[str, torch.Tensor | List[str]]:
        texts = [example.text for example in batch]
        sample_ids = torch.tensor([int(example.sample_id) for example in batch], dtype=torch.long)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        payload: Dict[str, torch.Tensor | List[str]] = {
            "sample_ids": sample_ids,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if batch[0].label is not None:
            payload["labels"] = torch.tensor([example.label for example in batch], dtype=torch.float32)
        return payload


class BinaryClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding).squeeze(-1)
        return logits


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def compute_binary_metrics(gold: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    if len(gold) != len(pred):
        raise ValueError("Gold labels and predicted labels have different lengths.")

    tp = fp = tn = fn = 0
    for gold_value, pred_value in zip(gold, pred):
        if gold_value == 1 and pred_value == 1:
            tp += 1
        elif gold_value == 0 and pred_value == 1:
            fp += 1
        elif gold_value == 0 and pred_value == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(gold) if gold else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def find_best_threshold(gold: Sequence[int], probabilities: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(gold, [1 if p >= 0.5 else 0 for p in probabilities])
    best_score = best_metrics["f1"]

    for step in range(5, 100, 5):
        threshold = step / 100.0
        pred = [1 if p >= threshold else 0 for p in probabilities]
        metrics = compute_binary_metrics(gold, pred)
        score = metrics["f1"]
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def run_validation(
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    threshold: float | None = None,
    progress_desc: str | None = None,
) -> Dict[str, object]:
    model.eval()
    losses: List[float] = []
    sample_ids: List[int] = []
    gold_labels: List[int] = []
    probabilities: List[float] = []

    with torch.no_grad():
        iterator: Iterable[Dict[str, torch.Tensor | List[str]]]
        if progress_desc is not None:
            iterator = tqdm(
                dataloader,
                desc=progress_desc,
                leave=False,
                disable=not accelerator.is_local_main_process,
            )
        else:
            iterator = dataloader

        for batch in iterator:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probs = torch.sigmoid(logits)
            gathered_probs = accelerator.gather_for_metrics(probs)
            gathered_sample_ids = accelerator.gather_for_metrics(batch["sample_ids"])
            probabilities.extend(float(x) for x in gathered_probs.detach().cpu().tolist())
            sample_ids.extend(int(x) for x in gathered_sample_ids.detach().cpu().tolist())
            if "labels" in batch:
                labels = batch["labels"]
                loss = criterion(logits, labels)
                gathered_labels = accelerator.gather_for_metrics(labels)
                gathered_loss = accelerator.gather_for_metrics(loss.detach().unsqueeze(0))
                gold_labels.extend(int(x) for x in gathered_labels.detach().cpu().tolist())
                losses.extend(float(x) for x in gathered_loss.detach().cpu().tolist())

    result: Dict[str, object] = {
        "sample_ids": sample_ids,
        "probabilities": probabilities,
        "loss": sum(losses) / len(losses) if losses else None,
    }

    if gold_labels:
        if threshold is None:
            threshold, metrics = find_best_threshold(gold_labels, probabilities)
        else:
            pred = [1 if p >= threshold else 0 for p in probabilities]
            metrics = compute_binary_metrics(gold_labels, pred)
        result["labels"] = gold_labels
        result["threshold"] = threshold
        result["metrics"] = metrics
        result["predictions"] = [1 if p >= threshold else 0 for p in probabilities]
    else:
        used_threshold = 0.5 if threshold is None else threshold
        result["threshold"] = used_threshold
        result["predictions"] = [1 if p >= used_threshold else 0 for p in probabilities]

    return result


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    accelerator: Accelerator,
    tokenizer,
    metadata: Dict[str, object],
) -> None:
    ensure_dir(output_dir)
    torch.save(accelerator.unwrap_model(model).state_dict(), output_dir / "best_model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    write_json(output_dir / "metadata.json", metadata)


def load_checkpoint(
    checkpoint_dir: Path,
    accelerator: Accelerator,
) -> Tuple[nn.Module, object, Dict[str, object]]:
    metadata = read_json(checkpoint_dir / "metadata.json")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    model = BinaryClassifier(
        model_name=str(metadata["model_name"]),
        dropout=float(metadata["dropout"]),
    )
    state_dict = torch.load(checkpoint_dir / "best_model.pt", map_location=accelerator.device)
    model.load_state_dict(normalize_state_dict_keys(state_dict))
    model.to(accelerator.device)
    return model, tokenizer, metadata


def create_rows_from_saved_split(train_rows: Sequence[Dict[str, str]], split_payload: Dict[str, object]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train_questions = set(split_payload["train_questions"])
    val_questions = set(split_payload["val_questions"])

    train_subset = [row for row in train_rows if row["question"] in train_questions]
    val_subset = [row for row in train_rows if row["question"] in val_questions]
    return train_subset, val_subset


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    train_path = Path(args.train_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    all_rows = read_tsv(train_path)
    train_rows, val_rows, split_payload = split_rows_by_question(
        all_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    write_json(output_dir / "split.json", split_payload)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = Task2Dataset(train_rows, include_labels=True)
    val_dataset = Task2Dataset(val_rows, include_labels=True)
    collator = Collator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    model = BinaryClassifier(model_name=args.model_name, dropout=args.dropout).to(device)

    num_pos = sum(label_from_row(row) for row in train_rows)
    num_neg = len(train_rows) - num_pos
    pos_weight_value = num_neg / max(1, num_pos)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_optimizer_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_f1 = -1.0
    best_metadata: Dict[str, object] | None = None

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "num_visible_gpus": torch.cuda.device_count() if device.type == "cuda" else 0,
                    "accelerate_processes": accelerator.num_processes,
                    "distributed_type": str(accelerator.distributed_type),
                    "num_train_rows": len(train_rows),
                    "num_val_rows": len(val_rows),
                    "num_train_batches": len(train_loader),
                    "num_val_batches": len(val_loader),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [train]",
            leave=True,
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar, start=1):
            with accelerator.accumulate(model):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = criterion(logits, batch["labels"])
                running_loss += float(loss.detach().item())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(
                loss=f"{running_loss / step:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        val_result = run_validation(
            accelerator=accelerator,
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            threshold=None,
            progress_desc=f"Epoch {epoch}/{args.epochs} [val]",
        )
        if accelerator.is_main_process:
            val_metrics = val_result["metrics"]
            threshold = float(val_result["threshold"])
            train_loss = running_loss / max(1, len(train_loader))
            val_loss = val_result["loss"]

            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "threshold": threshold,
            }
            print(json.dumps(epoch_summary, ensure_ascii=True), flush=True)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_metadata = {
                    "model_name": args.model_name,
                    "dropout": args.dropout,
                    "max_length": args.max_length,
                    "threshold": threshold,
                    "seed": args.seed,
                    "val_ratio": args.val_ratio,
                    "train_path": str(train_path),
                    "num_train_rows": len(train_rows),
                    "num_val_rows": len(val_rows),
                    "num_train_questions": len(split_payload["train_questions"]),
                    "num_val_questions": len(split_payload["val_questions"]),
                    "best_metrics": val_metrics,
                    "best_epoch": epoch,
                    "pos_weight": pos_weight_value,
                }
                save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    accelerator=accelerator,
                    tokenizer=tokenizer,
                    metadata=best_metadata,
                )
                save_prediction_rows(
                    output_path=output_dir / "val_predictions.tsv",
                    sample_ids=val_result["sample_ids"],
                    probabilities=val_result["probabilities"],
                    predictions=val_result["predictions"],
                    labels=val_result["labels"],
                )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if best_metadata is None:
            raise RuntimeError("Training ended without producing a checkpoint.")

        print(
            json.dumps(
                {"best_checkpoint_dir": str(output_dir), "best_metrics": best_metadata["best_metrics"]},
                ensure_ascii=True,
            ),
            flush=True,
        )


def save_prediction_rows(
    output_path: Path,
    sample_ids: Sequence[int],
    probabilities: Sequence[float],
    predictions: Sequence[int],
    labels: Sequence[int] | None = None,
) -> None:
    rows: List[Dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        row: Dict[str, object] = {
            "sample_id": sample_id,
            "probability": f"{probabilities[index]:.8f}",
            "prediction": predictions[index],
        }
        if labels is not None:
            row["label"] = labels[index]
        rows.append(row)

    fieldnames = ["sample_id", "probability", "prediction"]
    if labels is not None:
        fieldnames.append("label")
    write_tsv(output_path, rows, fieldnames)


def validate_command(args: argparse.Namespace) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    checkpoint_dir = Path(args.checkpoint_dir)

    model, tokenizer, metadata = load_checkpoint(checkpoint_dir=checkpoint_dir, accelerator=accelerator)
    train_rows = read_tsv(Path(args.train_path))
    split_payload = read_json(checkpoint_dir / "split.json")
    _, val_rows = create_rows_from_saved_split(train_rows, split_payload)

    val_dataset = Task2Dataset(val_rows, include_labels=True)
    collator = Collator(tokenizer=tokenizer, max_length=int(metadata["max_length"]))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    criterion = nn.BCEWithLogitsLoss()
    model, val_loader = accelerator.prepare(model, val_loader)

    result = run_validation(
        accelerator=accelerator,
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        threshold=float(metadata["threshold"]),
    )
    if accelerator.is_main_process:
        output_path = Path(args.output_path) if args.output_path else checkpoint_dir / "val_predictions.tsv"
        save_prediction_rows(
            output_path=output_path,
            sample_ids=result["sample_ids"],
            probabilities=result["probabilities"],
            predictions=result["predictions"],
            labels=result["labels"],
        )
        print(json.dumps({"validation_metrics": result["metrics"], "output_path": str(output_path)}, ensure_ascii=True))


def test_command(args: argparse.Namespace) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    checkpoint_dir = Path(args.checkpoint_dir)
    model, tokenizer, metadata = load_checkpoint(checkpoint_dir=checkpoint_dir, accelerator=accelerator)

    test_rows = read_tsv(Path(args.test_path))
    test_dataset = Task2Dataset(test_rows, include_labels=False)
    collator = Collator(tokenizer=tokenizer, max_length=int(metadata["max_length"]))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    criterion = nn.BCEWithLogitsLoss()
    model, test_loader = accelerator.prepare(model, test_loader)

    result = run_validation(
        accelerator=accelerator,
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        threshold=float(metadata["threshold"]),
    )
    if accelerator.is_main_process:
        output_path = Path(args.output_path)
        save_prediction_rows(
            output_path=output_path,
            sample_ids=result["sample_ids"],
            probabilities=result["probabilities"],
            predictions=result["predictions"],
            labels=None,
        )
        print(json.dumps({"test_output_path": str(output_path), "threshold": result["threshold"]}, ensure_ascii=True))


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "validate":
        validate_command(args)
    elif args.command == "test":
        test_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

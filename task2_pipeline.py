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


GRAPH_STATS_DIM = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run graph-aware transformer models for TextGraphs shared task 2."
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
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--node-max-length", type=int, default=16)
    train_parser.add_argument("--max-nodes", type=int, default=30)
    train_parser.add_argument("--freeze-embeddings", action="store_true")
    train_parser.add_argument("--freeze-layers", type=int, default=0)
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
    parser.add_argument(
        "--model-type",
        type=str,
        default="cross_attention",
        choices=["linearized_text", "graph_stats", "cross_attention"],
        help="Model family to train.",
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
            triples.append(f"{source_label} -> {str(edge['label']).strip()} -> {target_label}")

    if not triples:
        return "NO_GRAPH_EDGES"
    return " [GRAPH_SEP] ".join(triples)


def decorate_node_label(node: Dict[str, object]) -> str:
    node_type = str(node.get("type", "NODE"))
    label = str(node.get("label", "")).strip()
    if node_type == "QUESTIONS_ENTITY":
        prefix = "[QUESTION_ENTITY]"
    elif node_type == "ANSWER_CANDIDATE_ENTITY":
        prefix = "[ANSWER_ENTITY]"
    else:
        prefix = "[INTERNAL_ENTITY]"
    return f"{prefix} {label}".strip()


def extract_graph_stats(graph_value: object) -> List[float]:
    graph = parse_graph(graph_value)
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    num_nodes = len(nodes)
    num_edges = len(links)
    if num_nodes == 0:
        return [0.0] * GRAPH_STATS_DIM

    answer_nodes = [node for node in nodes if node.get("type") == "ANSWER_CANDIDATE_ENTITY"]
    question_nodes = [node for node in nodes if node.get("type") == "QUESTIONS_ENTITY"]
    answer_id = answer_nodes[0]["id"] if answer_nodes else -1
    question_ids = {node["id"] for node in question_nodes}

    answer_in = sum(1 for link in links if link.get("target") == answer_id)
    answer_out = sum(1 for link in links if link.get("source") == answer_id)
    answer_degree = answer_in + answer_out
    direct_connection = int(
        any(
            (link.get("source") in question_ids and link.get("target") == answer_id)
            or (link.get("target") in question_ids and link.get("source") == answer_id)
            for link in links
        )
    )
    num_question_entities = len(question_nodes)
    avg_path_proxy = num_nodes / max(1, num_question_entities)

    degrees = []
    for node in nodes:
        degree = sum(
            1
            for link in links
            if link.get("source") == node["id"] or link.get("target") == node["id"]
        )
        degrees.append(degree)
    max_degree = max(degrees) if degrees else 0
    max_edges = num_nodes * (num_nodes - 1)
    density = num_edges / max(1, max_edges)

    return [
        float(num_nodes),
        float(num_edges),
        float(answer_degree),
        float(answer_in),
        float(answer_out),
        float(direct_connection),
        float(num_question_entities),
        float(avg_path_proxy),
        float(density),
    ]


def extract_graph_node_labels(graph_value: object, max_nodes: int) -> Tuple[List[str], List[float]]:
    graph = parse_graph(graph_value)
    nodes = sorted(graph.get("nodes", []), key=lambda item: item.get("id", 0))[:max_nodes]
    labels = [decorate_node_label(node) for node in nodes]
    node_mask = [1.0] * len(labels)
    while len(labels) < max_nodes:
        labels.append("")
        node_mask.append(0.0)
    return labels, node_mask


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
    sample_id: int
    question: str
    answer: str
    linearized_text: str
    graph_stats: List[float]
    graph_node_labels: List[str]
    graph_node_mask: List[float]
    label: int | None


class Task2Dataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        include_labels: bool,
        max_nodes: int,
        stats_mean: torch.Tensor | None = None,
        stats_std: torch.Tensor | None = None,
    ) -> None:
        examples: List[Example] = []
        for row in rows:
            stats = torch.tensor(extract_graph_stats(row["graph"]), dtype=torch.float32)
            if stats_mean is not None and stats_std is not None:
                safe_std = stats_std.clamp(min=1e-8)
                stats = (stats - stats_mean) / safe_std
            node_labels, node_mask = extract_graph_node_labels(row["graph"], max_nodes=max_nodes)
            examples.append(
                Example(
                    sample_id=int(row["sample_id"]),
                    question=row["question"].strip(),
                    answer=row["answerEntity"].strip(),
                    linearized_text=build_linearized_input(row),
                    graph_stats=stats.tolist(),
                    graph_node_labels=node_labels,
                    graph_node_mask=node_mask,
                    label=label_from_row(row) if include_labels else None,
                )
            )
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def build_linearized_input(row: Dict[str, str]) -> str:
    parts = [
        f"Question: {row['question'].strip()}",
        f"Question entities: {row.get('questionEntity', '').strip()}",
        f"Candidate answer: {row['answerEntity'].strip()}",
        f"Graph: {linearize_graph(row['graph'])}",
    ]
    return " [SEP] ".join(parts)


def compute_stats_normalization(rows: Sequence[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    stats = torch.tensor([extract_graph_stats(row["graph"]) for row in rows], dtype=torch.float32)
    stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    return stats.mean(dim=0), stats.std(dim=0)


class Collator:
    def __init__(self, tokenizer, model_type: str, max_length: int, node_max_length: int) -> None:
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.node_max_length = node_max_length

    def __call__(self, batch: Sequence[Example]) -> Dict[str, torch.Tensor]:
        sample_ids = torch.tensor([example.sample_id for example in batch], dtype=torch.long)
        payload: Dict[str, torch.Tensor] = {"sample_ids": sample_ids}

        if self.model_type == "linearized_text":
            encoded = self.tokenizer(
                [example.linearized_text for example in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                [example.question for example in batch],
                [example.answer for example in batch],
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
        payload["input_ids"] = encoded["input_ids"]
        payload["attention_mask"] = encoded["attention_mask"]

        if self.model_type == "graph_stats":
            payload["graph_stats"] = torch.tensor([example.graph_stats for example in batch], dtype=torch.float32)
        elif self.model_type == "cross_attention":
            batch_size = len(batch)
            max_nodes = len(batch[0].graph_node_labels)
            flat_node_labels = [label for example in batch for label in example.graph_node_labels]
            node_encoded = self.tokenizer(
                flat_node_labels,
                padding="max_length",
                truncation=True,
                max_length=self.node_max_length,
                return_tensors="pt",
            )
            payload["node_input_ids"] = node_encoded["input_ids"].view(batch_size, max_nodes, -1)
            payload["node_attention_mask"] = node_encoded["attention_mask"].view(batch_size, max_nodes, -1)
            payload["node_mask"] = torch.tensor([example.graph_node_mask for example in batch], dtype=torch.float32)

        if batch[0].label is not None:
            payload["labels"] = torch.tensor([example.label for example in batch], dtype=torch.float32)

        return payload


def freeze_encoder_layers(model: AutoModel, freeze_embeddings: bool, freeze_layers: int) -> None:
    if freeze_embeddings and hasattr(model, "embeddings"):
        for parameter in model.embeddings.parameters():
            parameter.requires_grad = False
    if freeze_layers > 0 and hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        for layer in model.encoder.layer[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False


class BaseClassifier(nn.Module):
    def __init__(self, model_name: str, freeze_embeddings: bool, freeze_layers: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        freeze_encoder_layers(self.encoder, freeze_embeddings, freeze_layers)
        self.hidden_size = self.encoder.config.hidden_size

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state[:, 0, :]


class LinearizedTextClassifier(BaseClassifier):
    def __init__(self, model_name: str, dropout: float, freeze_embeddings: bool, freeze_layers: int) -> None:
        super().__init__(model_name=model_name, freeze_embeddings=freeze_embeddings, freeze_layers=freeze_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encode_text(input_ids, attention_mask)
        return self.classifier(cls).squeeze(-1)


class GraphStatsClassifier(BaseClassifier):
    def __init__(self, model_name: str, dropout: float, freeze_embeddings: bool, freeze_layers: int) -> None:
        super().__init__(model_name=model_name, freeze_embeddings=freeze_embeddings, freeze_layers=freeze_layers)
        self.stats_proj = nn.Sequential(
            nn.Linear(GRAPH_STATS_DIM, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_stats: torch.Tensor,
    ) -> torch.Tensor:
        cls = self.encode_text(input_ids, attention_mask)
        stats = self.stats_proj(graph_stats)
        return self.classifier(torch.cat([cls, stats], dim=-1)).squeeze(-1)


class CrossAttentionClassifier(BaseClassifier):
    def __init__(self, model_name: str, dropout: float, freeze_embeddings: bool, freeze_layers: int) -> None:
        super().__init__(model_name=model_name, freeze_embeddings=freeze_embeddings, freeze_layers=freeze_layers)
        self.node_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode_node_labels(
        self,
        node_input_ids: torch.Tensor,
        node_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_nodes, seq_len = node_input_ids.shape
        flat_input_ids = node_input_ids.view(batch_size * max_nodes, seq_len)
        flat_attention_mask = node_attention_mask.view(batch_size * max_nodes, seq_len).float()
        embedding_layer = self.encoder.get_input_embeddings()
        token_embeddings = embedding_layer(flat_input_ids)
        denom = flat_attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (token_embeddings * flat_attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled.view(batch_size, max_nodes, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_input_ids: torch.Tensor,
        node_attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = outputs.last_hidden_state[:, 0:1, :]

        node_feats = self.encode_node_labels(node_input_ids=node_input_ids, node_attention_mask=node_attention_mask)
        key_values = self.node_proj(node_feats)

        key_padding_mask = node_mask == 0
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        attended, _ = self.cross_attn(
            query=cls,
            key=key_values,
            value=key_values,
            key_padding_mask=key_padding_mask,
        )
        attended = self.layer_norm(attended.squeeze(1) + cls.squeeze(1))
        return self.classifier(torch.cat([cls.squeeze(1), attended], dim=-1)).squeeze(-1)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model_type == "linearized_text":
        return LinearizedTextClassifier(
            model_name=args.model_name,
            dropout=args.dropout,
            freeze_embeddings=args.freeze_embeddings,
            freeze_layers=args.freeze_layers,
        )
    if args.model_type == "graph_stats":
        return GraphStatsClassifier(
            model_name=args.model_name,
            dropout=args.dropout,
            freeze_embeddings=args.freeze_embeddings,
            freeze_layers=args.freeze_layers,
        )
    if args.model_type == "cross_attention":
        return CrossAttentionClassifier(
            model_name=args.model_name,
            dropout=args.dropout,
            freeze_embeddings=args.freeze_embeddings,
            freeze_layers=args.freeze_layers,
        )
    raise ValueError(f"Unsupported model type: {args.model_type}")


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
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def find_best_threshold(gold: Sequence[int], probabilities: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(gold, [1 if p >= 0.5 else 0 for p in probabilities])
    best_score = best_metrics["f1"]
    for step in range(5, 100, 5):
        threshold = step / 100.0
        pred = [1 if p >= threshold else 0 for p in probabilities]
        metrics = compute_binary_metrics(gold, pred)
        if metrics["f1"] > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = metrics["f1"]
    return best_threshold, best_metrics


def forward_model(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    model_inputs = {key: value for key, value in batch.items() if key not in {"sample_ids", "labels"}}
    return model(**model_inputs)


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
        iterator: Iterable[Dict[str, torch.Tensor]]
        if progress_desc is not None:
            iterator = tqdm(dataloader, desc=progress_desc, leave=False, disable=not accelerator.is_local_main_process)
        else:
            iterator = dataloader

        for batch in iterator:
            logits = forward_model(model, batch)
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
            metrics = compute_binary_metrics(gold_labels, [1 if p >= threshold else 0 for p in probabilities])
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


def load_checkpoint(checkpoint_dir: Path, accelerator: Accelerator) -> Tuple[nn.Module, object, Dict[str, object]]:
    metadata = read_json(checkpoint_dir / "metadata.json")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    namespace = argparse.Namespace(**metadata)
    model = build_model(namespace)
    state_dict = torch.load(checkpoint_dir / "best_model.pt", map_location=accelerator.device)
    model.load_state_dict(normalize_state_dict_keys(state_dict))
    model.to(accelerator.device)
    return model, tokenizer, metadata


def create_rows_from_saved_split(
    train_rows: Sequence[Dict[str, str]],
    split_payload: Dict[str, object],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train_questions = set(split_payload["train_questions"])
    val_questions = set(split_payload["val_questions"])
    return (
        [row for row in train_rows if row["question"] in train_questions],
        [row for row in train_rows if row["question"] in val_questions],
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


def build_datasets(args: argparse.Namespace, train_rows: Sequence[Dict[str, str]], val_rows: Sequence[Dict[str, str]]) -> Tuple[Task2Dataset, Task2Dataset, torch.Tensor | None, torch.Tensor | None]:
    stats_mean = stats_std = None
    if args.model_type == "graph_stats":
        stats_mean, stats_std = compute_stats_normalization(train_rows)
    train_dataset = Task2Dataset(
        train_rows,
        include_labels=True,
        max_nodes=args.max_nodes,
        stats_mean=stats_mean,
        stats_std=stats_std,
    )
    val_dataset = Task2Dataset(
        val_rows,
        include_labels=True,
        max_nodes=args.max_nodes,
        stats_mean=stats_mean,
        stats_std=stats_std,
    )
    return train_dataset, val_dataset, stats_mean, stats_std


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
    train_rows, val_rows, split_payload = split_rows_by_question(all_rows, val_ratio=args.val_ratio, seed=args.seed)
    write_json(output_dir / "split.json", split_payload)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, val_dataset, _, _ = build_datasets(args, train_rows, val_rows)
    collator = Collator(
        tokenizer=tokenizer,
        model_type=args.model_type,
        max_length=args.max_length,
        node_max_length=args.node_max_length,
    )
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

    model = build_model(args).to(device)
    num_pos = sum(label_from_row(row) for row in train_rows)
    num_neg = len(train_rows) - num_pos
    pos_weight_value = num_neg / max(1, num_pos)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_optimizer_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_optimizer_steps * args.warmup_ratio),
        num_training_steps=total_optimizer_steps,
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "model_type": args.model_type,
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

    best_f1 = -1.0
    best_metadata: Dict[str, object] | None = None

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
                logits = forward_model(model, batch)
                loss = criterion(logits, batch["labels"])
                running_loss += float(loss.detach().item())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(loss=f"{running_loss / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

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
            epoch_summary = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_loader)),
                "val_loss": val_result["loss"],
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
                    "model_type": args.model_type,
                    "dropout": args.dropout,
                    "max_length": args.max_length,
                    "node_max_length": args.node_max_length,
                    "max_nodes": args.max_nodes,
                    "freeze_embeddings": args.freeze_embeddings,
                    "freeze_layers": args.freeze_layers,
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
                save_checkpoint(output_dir=output_dir, model=model, accelerator=accelerator, tokenizer=tokenizer, metadata=best_metadata)
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
        print(json.dumps({"best_checkpoint_dir": str(output_dir), "best_metrics": best_metadata["best_metrics"]}, ensure_ascii=True), flush=True)


def build_eval_dataset(rows: Sequence[Dict[str, str]], metadata: Dict[str, object], include_labels: bool) -> Task2Dataset:
    args = argparse.Namespace(**metadata)
    stats_mean = stats_std = None
    if args.model_type == "graph_stats":
        train_rows = read_tsv(Path(metadata["train_path"]))
        train_subset, _ = create_rows_from_saved_split(train_rows, read_json(Path(metadata["output_dir"]) / "split.json")) if "output_dir" in metadata else (train_rows, [])
        stats_mean, stats_std = compute_stats_normalization(train_subset)
    return Task2Dataset(
        rows,
        include_labels=include_labels,
        max_nodes=int(metadata["max_nodes"]),
        stats_mean=stats_mean,
        stats_std=stats_std,
    )


def validate_command(args: argparse.Namespace) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    checkpoint_dir = Path(args.checkpoint_dir)
    model, tokenizer, metadata = load_checkpoint(checkpoint_dir=checkpoint_dir, accelerator=accelerator)
    train_rows = read_tsv(Path(args.train_path))
    split_payload = read_json(checkpoint_dir / "split.json")
    train_subset, val_rows = create_rows_from_saved_split(train_rows, split_payload)

    stats_mean = stats_std = None
    if metadata["model_type"] == "graph_stats":
        stats_mean, stats_std = compute_stats_normalization(train_subset)

    val_dataset = Task2Dataset(
        val_rows,
        include_labels=True,
        max_nodes=int(metadata["max_nodes"]),
        stats_mean=stats_mean,
        stats_std=stats_std,
    )
    collator = Collator(
        tokenizer=tokenizer,
        model_type=str(metadata["model_type"]),
        max_length=int(metadata["max_length"]),
        node_max_length=int(metadata["node_max_length"]),
    )
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
        save_prediction_rows(output_path=output_path, sample_ids=result["sample_ids"], probabilities=result["probabilities"], predictions=result["predictions"], labels=result["labels"])
        print(json.dumps({"validation_metrics": result["metrics"], "output_path": str(output_path)}, ensure_ascii=True))


def test_command(args: argparse.Namespace) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    checkpoint_dir = Path(args.checkpoint_dir)
    model, tokenizer, metadata = load_checkpoint(checkpoint_dir=checkpoint_dir, accelerator=accelerator)
    test_rows = read_tsv(Path(args.test_path))

    train_rows = read_tsv(Path(metadata["train_path"])) if metadata["model_type"] == "graph_stats" else []
    stats_mean = stats_std = None
    if metadata["model_type"] == "graph_stats":
        train_subset, _ = create_rows_from_saved_split(train_rows, read_json(checkpoint_dir / "split.json"))
        stats_mean, stats_std = compute_stats_normalization(train_subset)

    test_dataset = Task2Dataset(
        test_rows,
        include_labels=False,
        max_nodes=int(metadata["max_nodes"]),
        stats_mean=stats_mean,
        stats_std=stats_std,
    )
    collator = Collator(
        tokenizer=tokenizer,
        model_type=str(metadata["model_type"]),
        max_length=int(metadata["max_length"]),
        node_max_length=int(metadata["node_max_length"]),
    )
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
        save_prediction_rows(output_path=output_path, sample_ids=result["sample_ids"], probabilities=result["probabilities"], predictions=result["predictions"])
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

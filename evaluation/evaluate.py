import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help=(
            "Path to a TSV file with predicted labels. Predicted labels must be "
            "stored in the 'prediction' column as binary labels: 0 or 1."
        ),
    )
    parser.add_argument(
        "--gold_labels_path",
        type=str,
        required=True,
        help=(
            "Path to a TSV file with ground truth labels. Ground truth labels must be "
            "stored in the 'correct' column as 'False' and 'True' strings."
        ),
    )
    return parser.parse_args()


def read_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(true_labels, pred_labels):
    if len(true_labels) != len(pred_labels):
        raise ValueError(
            "The number of predictions does not match the number of gold labels: "
            f"{len(pred_labels)} vs {len(true_labels)}."
        )

    tp = fp = tn = fn = 0
    for gold, pred in zip(true_labels, pred_labels):
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        else:
            fn += 1

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, len(true_labels))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main(args):
    predictions_rows = read_tsv(args.predictions_path)
    gold_rows = read_tsv(args.gold_labels_path)

    if not predictions_rows:
        raise RuntimeError("Prediction file is empty.")
    if not gold_rows:
        raise RuntimeError("Gold label file is empty.")
    if "prediction" not in predictions_rows[0]:
        raise RuntimeError("The 'prediction' column is not found in the prediction file.")
    if "correct" not in gold_rows[0]:
        raise RuntimeError("The 'correct' column is not found in the gold file.")
    if "sample_id" not in predictions_rows[0]:
        raise RuntimeError("The 'sample_id' column is not found in the prediction file.")
    if "sample_id" not in gold_rows[0]:
        raise RuntimeError("The 'sample_id' column is not found in the gold file.")

    pred_by_id = {}
    for row in predictions_rows:
        sample_id = row["sample_id"]
        pred_by_id[sample_id] = int(row["prediction"])

    pred_labels = list(pred_by_id.values())
    invalid_values = sorted({value for value in pred_labels if value not in (0, 1)})
    if invalid_values:
        raise RuntimeError(
            "Prediction file contains invalid labels. Expected only 0/1, found: "
            f"{invalid_values}"
        )

    aligned_gold_rows = [row for row in gold_rows if row["sample_id"] in pred_by_id]
    missing_predictions = [row["sample_id"] for row in gold_rows if row["sample_id"] not in pred_by_id]
    if missing_predictions and len(aligned_gold_rows) != len(gold_rows):
        print(
            "Warning: prediction file does not cover the entire gold file. "
            f"Scoring on {len(aligned_gold_rows)} overlapping rows."
        )

    true_labels = [1 if row["correct"] == "True" else 0 for row in aligned_gold_rows]
    pred_labels = [pred_by_id[row["sample_id"]] for row in aligned_gold_rows]
    metrics = compute_metrics(true_labels, pred_labels)

    print("Evaluation\n")
    print(f"\tPrecision: {metrics['precision']:.6f}")
    print(f"\tRecall: {metrics['recall']:.6f}")
    print(f"\tF1: {metrics['f1']:.6f}")
    print(f"\tAccuracy: {metrics['accuracy']:.6f}")


if __name__ == "__main__":
    main(parse_args())

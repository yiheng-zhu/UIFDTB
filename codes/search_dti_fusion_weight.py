"""Ensemble five DTI runs per prediction branch, then fuse branch probabilities.

Inputs are produced by generate_dti_predictions.py (mode=0 and mode=1).
This script does not load any model or rerun inference.
"""

import argparse
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score


def read_result(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")
    data = np.loadtxt(path, dtype=float, ndmin=2)
    if data.shape[1] != 2:
        raise ValueError(f"Expected two columns (label probability) in {path}; got {data.shape[1]}")
    labels, probabilities = data[:, 0], data[:, 1]
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite value found in {path}")
    return probabilities, labels


def save_result(labels, predictions, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for label, prediction in zip(labels, predictions):
            file.write(f"{label} {prediction}\n")


def get_average_result(workdir, mode, split, rounds):
    """Average probabilities for matching sample rows over five independent rounds."""
    predictions = []
    reference_labels = None
    for round_index in range(1, rounds + 1):
        path = os.path.join(
            workdir, "result", f"mode{mode}", f"round{round_index}",
            split, "result1.txt"
        )
        probabilities, labels = read_result(path)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise ValueError(
                f"Labels or sample order differ across runs: {path}. "
                "All five rounds must have identical sample order."
            )
        predictions.append(probabilities)
    return np.mean(np.stack(predictions, axis=0), axis=0), reference_labels


def probability_metrics(labels, probabilities):
    auroc = roc_auc_score(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    auprc = metrics.auc(recall, precision)
    return float(auroc), float(auprc)


def test_metrics(labels, probabilities):
    """Same core metrics as search_dti_fusion_weight.py.

    The source code's threshold indexing can go out of bounds. Choose the
    corresponding threshold from the valid threshold array instead.
    """
    auroc, auprc = probability_metrics(labels, probabilities)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-6)
    if len(thresholds) == 0:
        raise ValueError("No decision thresholds available for test predictions")
    start = 5 if len(thresholds) > 5 else 0
    threshold_index = start + int(np.argmax(f1_curve[start:len(thresholds)]))
    threshold = float(thresholds[threshold_index])
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": (tn + tp) / total if total else float('nan'),
        "Sensitivity": tp / (tp + fn) if tp + fn else float('nan'),
        "Specificity": tn / (tn + fp) if tn + fp else float('nan'),
        "MCC": float(metrics.matthews_corrcoef(labels, predicted)),
        "F1": float(metrics.f1_score(labels, predicted, zero_division=0)),
        "Threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Average DTI mode0/mode1 probabilities across five rounds, then optimize fusion weight."
    )
    parser.add_argument("workdir", help="Dataset workdir containing result/mode0 and result/mode1")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be a positive integer")
    workdir = os.path.abspath(args.workdir)
    if not os.path.isdir(workdir):
        raise NotADirectoryError(workdir)

    averages = {}
    for mode in (0, 1):
        averages[mode] = {}
        for split in ("valid", "test"):
            probabilities, labels = get_average_result(
                workdir, mode, split, args.rounds
            )
            averages[mode][split] = (probabilities, labels)
            output = os.path.join(
                workdir, "result", f"mode{mode}", "final_result", split,
                "result1.txt"
            )
            save_result(labels, probabilities, output)
            auroc, auprc = probability_metrics(labels, probabilities)
            print(f"[Average] mode={mode} {split}: AUROC={auroc:.6f}, AUPRC={auprc:.6f}")
            print(f"          Saved: {output}")

    for split in ("valid", "test"):
        labels0 = averages[0][split][1]
        labels1 = averages[1][split][1]
        if not np.array_equal(labels0, labels1):
            raise ValueError(f"Labels or sample order differ between mode0 and mode1 on {split}")

    valid0, labels_valid = averages[0]["valid"]
    valid1, _ = averages[1]["valid"]
    best_weight, best_auroc = None, -np.inf
    weight_records = []
    for step in range(1, 101):
        w = step / 100.0
        merged = w * valid0 + (1.0 - w) * valid1
        auroc, auprc = probability_metrics(labels_valid, merged)
        weight_records.append((w, auroc, auprc))
        print(f"w={w:.2f} | Validation AUROC={auroc:.6f} | AUPRC={auprc:.6f}")
        if auroc > best_auroc:
            best_weight, best_auroc = w, auroc

    print(f"\nBest weight (validation AUROC): {best_weight:.2f}")
    combined_dir = os.path.join(workdir, "result", "combine_result")
    valid_merged = best_weight * valid0 + (1.0 - best_weight) * valid1
    valid_auroc, valid_auprc = probability_metrics(labels_valid, valid_merged)
    save_result(labels_valid, valid_merged, os.path.join(combined_dir, "valid", "result.txt"))

    test0, labels_test = averages[0]["test"]
    test1, _ = averages[1]["test"]
    test_merged = best_weight * test0 + (1.0 - best_weight) * test1
    save_result(labels_test, test_merged, os.path.join(combined_dir, "test", "result.txt"))
    scores = test_metrics(labels_test, test_merged)
    print("[Combined Validation] "
          f"AUROC={valid_auroc:.6f}, AUPRC={valid_auprc:.6f}")
    print("[Combined Test] " + " | ".join(f"{k}={v:.6f}" for k, v in scores.items()))

    record_path = os.path.join(combined_dir, "final_record.txt")
    os.makedirs(combined_dir, exist_ok=True)
    with open(record_path, "w", encoding="utf-8") as file:
        file.write("Prediction source filename: result1.txt\n")
        file.write(f"Number of training rounds: {args.rounds}\n\n")
        for w, auroc, auprc in weight_records:
            file.write(f"w={w:.2f} Validation AUROC={auroc:.6f} AUPRC={auprc:.6f}\n")
        file.write(f"\nBest weight (validation AUROC): {best_weight:.2f}\n")
        file.write(f"Best validation AUROC: {valid_auroc:.6f}\n")
        file.write(f"Best validation AUPRC: {valid_auprc:.6f}\n")
        file.write("\nTest metrics at best validation weight:\n")
        for name, value in scores.items():
            file.write(f"{name}: {value:.6f}\n")
    print(f"Saved combined predictions to: {combined_dir}")
    print(f"Saved evaluation record to: {record_path}")


if __name__ == "__main__":
    main()

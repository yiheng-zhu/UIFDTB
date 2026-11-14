import os
import sys

# Set CUDA visibility before importing PyTorch and DGL.
if len(sys.argv) >= 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

import pickle as pkl
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn import metrics

from utils import *
from dti_model import *
from config import dataset_name, graph_name


def read_pkl(file_path):
    with open(file_path, "rb") as f:
        return pkl.load(f)


def cl_loss_pos_with_neg(z_a_pos, z_b_pos, z_b_neg=None, tau=0.07):
    positive_count = z_a_pos.size(0)
    logits_pos = (z_a_pos @ z_b_pos.t()) / tau
    if z_b_neg is not None and z_b_neg.size(0) > 0:
        logits = torch.cat(
            [logits_pos, (z_a_pos @ z_b_neg.t()) / tau], dim=1
        )
    else:
        logits = logits_pos
    labels = torch.arange(positive_count, device=z_a_pos.device)
    return F.cross_entropy(logits, labels)


def cl_loss_bidirectional_pos_with_neg(z_d, z_t, y, tau=0.07):
    y = y.view(-1)
    positive_mask = y == 1
    negative_mask = y == 0
    z_d_pos, z_t_pos = z_d[positive_mask], z_t[positive_mask]
    z_d_neg, z_t_neg = z_d[negative_mask], z_t[negative_mask]
    if z_d_pos.size(0) < 2:
        return None
    loss_dt = cl_loss_pos_with_neg(z_d_pos, z_t_pos, z_t_neg, tau)
    loss_td = cl_loss_pos_with_neg(z_t_pos, z_d_pos, z_d_neg, tau)
    return 0.5 * (loss_dt + loss_td)


TAU = 0.07
LAMBDA_CL = 0.001
loss_fn = nn.BCELoss()


def train(model, device, train_loader, optimizer, epoch, graphs):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        batch_graphs = [graphs[i.item()] for i in data.pro]
        batch_graphs = dgl.batch(batch_graphs).to(device)
        graph_features = batch_graphs.ndata["x"]
        data = data.to(device)
        optimizer.zero_grad()

        output, z_d, z_t = model(data, batch_graphs, graph_features)
        labels = data.y.view(-1, 1).float().to(device)
        task_loss = loss_fn(output, labels)
        contrastive_loss = cl_loss_bidirectional_pos_with_neg(
            z_d=z_d, z_t=z_t, y=data.y, tau=TAU
        )
        loss = (
            task_loss if contrastive_loss is None
            else task_loss + LAMBDA_CL * contrastive_loss
        )
        loss.backward()
        optimizer.step()

        if batch_idx % 32 == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * train_loader.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / max(len(train_loader), 1),
                    loss.item()
                )
            )


def predicting(model, device, loader, graphs):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            batch_graphs = [graphs[i.item()] for i in data.pro]
            batch_graphs = dgl.batch(batch_graphs).to(device)
            graph_features = batch_graphs.ndata["x"]
            data = data.to(device)
            output, _, _ = model(data, batch_graphs, graph_features)
            total_preds = torch.cat((total_preds, output.cpu()), dim=0)
            total_labels = torch.cat(
                (total_labels, data.y.view(-1, 1).cpu()), dim=0
            )
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def save_result(labels, predictions, result_file):
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        for label, prediction in zip(labels, predictions):
            f.write(f"{label} {prediction}\n")


def evaluate_binary_classification(labels, predictions):
    auroc = roc_auc_score(labels, predictions)
    precision, recall, _ = precision_recall_curve(labels, predictions)
    auprc = metrics.auc(recall, precision)

    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    f1_values = 2 * precision * recall / (precision + recall + 1e-6)
    if len(thresholds) > 5:
        threshold = thresholds[5:][np.argmax(f1_values[5:])]
    elif len(thresholds) > 0:
        threshold = thresholds[np.argmax(f1_values[:len(thresholds)])]
    else:
        threshold = 0.5

    predicted = (np.asarray(predictions) >= threshold).astype(int)
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": (tn + tp) / max(tn + fp + fn + tp, 1),
        "Sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "Specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "MCC": metrics.matthews_corrcoef(labels, predicted),
        "F1": metrics.f1_score(labels, predicted, zero_division=0),
    }


def main(workdir, round_index):
    workdir = os.path.abspath(workdir)
    dataset = os.path.basename(os.path.normpath(workdir))

    graph_file = os.path.join(workdir, graph_name + ".pkl")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(
            f"Protein graph file not found: {graph_file}. "
            f"Please generate {graph_name}.pkl before training."
        )
    print("Loading protein graphs from:", graph_file)
    graphs = read_pkl(graph_file)

    modeling = [QSI]
    cuda_name = "cuda:0"
    num_epochs = 50
    train_batch_size = 64
    test_batch_size = 64
    learning_rate = 0.00003

    print("Dataset:", dataset)
    print("Working directory:", workdir)
    print("Device:", cuda_name if torch.cuda.is_available() else "cpu")
    print("Learning rate:", learning_rate)
    print("Epochs:", num_epochs)

    processed_files = {
        "train": os.path.join(workdir, dataset_name, "train.pt"),
        "val": os.path.join(workdir, dataset_name, "val.pt"),
        "test": os.path.join(workdir, dataset_name, "test.pt"),
    }
    missing = [p for p in processed_files.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Processed datasets are missing. Please run data_process.py first:\n"
            + "\n".join("  - " + p for p in missing)
        )

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    for model_class in modeling:
        model_name = model_class.__name__
        print("\nRunning", model_name, "on", dataset)

        train_data = MyDataset(
            root=workdir, workdir=workdir,
            dataset=os.path.join(dataset_name, "train.pt")
        )
        val_data = MyDataset(
            root=workdir, workdir=workdir,
            dataset=os.path.join(dataset_name, "val.pt")
        )
        test_data = MyDataset(
            root=workdir, workdir=workdir,
            dataset=os.path.join(dataset_name, "test.pt")
        )

        train_loader = DataLoader(
            train_data, batch_size=train_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=train_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=test_batch_size, shuffle=False
        )

        model = model_class().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=0.000001
        )
        best_auroc = 0.0
        best_epoch = 0

        round_res_dir = os.path.join(workdir, "result", f"round{round_index}")
        os.makedirs(round_res_dir, exist_ok=True)
        record_file = os.path.join(round_res_dir, "record.txt")

        with open(record_file, "w", encoding="utf-8") as record:
            for epoch in range(num_epochs):
                train(
                    model, device, train_loader, optimizer, epoch + 1, graphs
                )

                checkpoint_dir = os.path.join(
                    workdir, "model", f"round{round_index}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
                )

                # Validation
                val_labels, val_predictions = predicting(
                    model, device, val_loader, graphs
                )
                valid_file = os.path.join(
                    workdir, "result", f"round{round_index}", "valid",
                    f"result{epoch + 1}.txt"
                )
                save_result(val_labels, val_predictions, valid_file)
                val_metrics = evaluate_binary_classification(
                    val_labels, val_predictions
                )
                if val_metrics["AUROC"] > best_auroc:
                    best_auroc = val_metrics["AUROC"]
                    best_epoch = epoch

                val_message = (
                    f"At epoch: {epoch + 1} AUROC {val_metrics['AUROC']} ; "
                    f"AUPRC {val_metrics['AUPRC']}"
                )
                print(val_message)
                record.write(val_message + "\n")

                # Test
                test_labels, test_predictions = predicting(
                    model, device, test_loader, graphs
                )
                test_file = os.path.join(
                    workdir, "result", f"round{round_index}", "test",
                    f"result{epoch + 1}.txt"
                )
                save_result(test_labels, test_predictions, test_file)
                test_metrics = evaluate_binary_classification(
                    test_labels, test_predictions
                )
                test_message = (
                    f"Test at epoch: {epoch + 1}, "
                    f"AUROC {test_metrics['AUROC']} ; "
                    f"AUPRC {test_metrics['AUPRC']} ; "
                    f"Accuracy {test_metrics['Accuracy']} ; "
                    f"Sensitivity {test_metrics['Sensitivity']} ; "
                    f"Specificity {test_metrics['Specificity']} ; "
                    f"MCC {test_metrics['MCC']} ; F1 {test_metrics['F1']}"
                )
                print("***")
                print(test_message)
                print("***")
                record.write(test_message + "\n\n")
                record.flush()

            record.write(f"Best epoch {best_epoch + 1}\n")
            record.write(f"Best validation AUROC {best_auroc}\n")

        print(
            f"Round {round_index} completed. Best epoch: {best_epoch + 1}; "
            f"best validation AUROC: {best_auroc}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python train_dti.py <workdir> <gpu_id>\n"
            "Example: python train_dti.py ./test_example/ 0"
        )

    workdir = os.path.abspath(sys.argv[1])
    gpu_id = sys.argv[2]
    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"Working directory not found: {workdir}")

    # Preserve the original DTI setting: 10 independent rounds.
    for round_index in range(1, 6):
        main(workdir, round_index)

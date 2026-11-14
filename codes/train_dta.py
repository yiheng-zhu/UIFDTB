import os
import sys

# Set the visible GPU before importing PyTorch and DGL.
if len(sys.argv) >= 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

import pickle as pkl
from math import sqrt

import dgl
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from utils import *
from dta_model import *
from config import dataset_name, graph_name


def read_pkl(file_path):
    with open(file_path, "rb") as f:
        return pkl.load(f)


def train(model, device, train_loader, optimizer, epoch, graphs):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()

    for batch_idx, data in enumerate(train_loader):
        batch_indices = data.pro
        batch_graphs = [graphs[i.item()] for i in batch_indices]
        batch_graphs = dgl.batch(batch_graphs).to(device)
        graph_features = batch_graphs.ndata["x"]

        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, batch_graphs, graph_features)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()

        if batch_idx % 32 == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / max(len(train_loader), 1),
                    loss.item(),
                )
            )


def mse(y, prediction):
    return ((y - prediction) ** 2).mean(axis=0)


def get_cindex(y, prediction):
    y = np.asarray(y).ravel()
    prediction = np.asarray(prediction).ravel()

    y_diff = y[:, None] - y[None, :]
    prediction_diff = prediction[:, None] - prediction[None, :]
    mask = y_diff > 0

    if np.sum(mask) == 0:
        return np.nan

    score = (prediction_diff > 0).astype(float)
    score += 0.5 * (prediction_diff == 0).astype(float)
    return np.sum(score[mask]) / np.sum(mask)


def r_squared_error(y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    y_obs_centered = y_obs - np.mean(y_obs)
    y_pred_centered = y_pred - np.mean(y_pred)

    numerator = np.sum(y_pred_centered * y_obs_centered) ** 2
    denominator = (
        np.sum(y_obs_centered ** 2) * np.sum(y_pred_centered ** 2)
    )

    if denominator == 0:
        return np.nan
    return numerator / float(denominator)


def get_k(y_obs, y_pred):
    denominator = np.sum(np.asarray(y_pred) ** 2)
    if denominator == 0:
        return np.nan
    return np.sum(np.asarray(y_obs) * np.asarray(y_pred)) / float(denominator)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    if not np.isfinite(k):
        return np.nan

    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)
    denominator = np.sum((y_obs - np.mean(y_obs)) ** 2)

    if denominator == 0:
        return np.nan

    numerator = np.sum((y_obs - k * y_pred) ** 2)
    return 1.0 - numerator / float(denominator)


def get_rm2(y_obs, y_pred):
    r2 = r_squared_error(y_obs, y_pred)
    r02 = squared_error_zero(y_obs, y_pred)

    if not np.isfinite(r2) or not np.isfinite(r02):
        return np.nan

    return r2 * (1.0 - np.sqrt(np.abs(r2 - r02)))


def predicting(model, device, loader, graphs):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    print("Make prediction for {} samples...".format(len(loader.dataset)))

    with torch.no_grad():
        for data in loader:
            batch_indices = data.pro
            batch_graphs = [graphs[i.item()] for i in batch_indices]
            batch_graphs = dgl.batch(batch_graphs).to(device)
            graph_features = batch_graphs.ndata["x"]

            data = data.to(device)
            output = model(data, batch_graphs, graph_features)
            output = output.clamp(min=0.0)

            total_preds = torch.cat((total_preds, output.cpu()), dim=0)
            total_labels = torch.cat(
                (total_labels, data.y.view(-1, 1).cpu()), dim=0
            )

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


loss_fn = nn.MSELoss()


def save_result(labels, predictions, result_file):
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        for label, prediction in zip(labels, predictions):
            f.write(f"{label} {prediction}\n")


def main(workdir, round_index):
    workdir = os.path.abspath(workdir)
    dataset = os.path.basename(os.path.normpath(workdir))

    # Only graphs.pkl is used.
    graph_file = os.path.join(workdir, graph_name + ".pkl")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(
            f"Protein graph file not found: {graph_file}. "
            "Please generate graphs.pkl before training."
        )

    print("Loading protein graphs from:", graph_file)
    graphs = read_pkl(graph_file)

    modeling = [QSI]
    cuda_name = "cuda:0"
    num_epochs = 100
    train_batch_size = 64
    test_batch_size = 64
    learning_rate = 0.00003

    print("Dataset:", dataset)
    print("Working directory:", workdir)
    print("Device:", cuda_name if torch.cuda.is_available() else "cpu")
    print("Learning rate:", learning_rate)
    print("Epochs:", num_epochs)

    # Processed PyTorch datasets are stored in workdir/dataset_name/.
    processed_files = {
        "train": os.path.join(workdir, dataset_name, "train.pt"),
        "val": os.path.join(workdir, dataset_name, "val.pt"),
        "test": os.path.join(workdir, dataset_name, "test.pt"),
    }
    missing_files = [
        path for path in processed_files.values()
        if not os.path.isfile(path)
    ]
    if missing_files:
        raise FileNotFoundError(
            "Processed datasets are missing. Please run data_process.py first:\n"
            + "\n".join("  - " + path for path in missing_files)
        )

    for model_class in modeling:
        model_name = model_class.__name__
        print("\nRunning", model_name, "on", dataset)

        train_data = MyDataset(
            root=workdir,
            workdir=workdir,
            dataset=os.path.join(dataset_name, "train.pt"),
        )
        val_data = MyDataset(
            root=workdir,
            workdir=workdir,
            dataset=os.path.join(dataset_name, "val.pt"),
        )
        test_data = MyDataset(
            root=workdir,
            workdir=workdir,
            dataset=os.path.join(dataset_name, "test.pt"),
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

        device = torch.device(
            cuda_name if torch.cuda.is_available() else "cpu"
        )
        model = model_class().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.000001,
        )

        best_mse = float("inf")
        best_epoch = 0

        # res/roundN/ directly; no dataset-name subdirectory.
        round_res_dir = os.path.join(
            workdir, "result", f"round{round_index}"
        )
        os.makedirs(round_res_dir, exist_ok=True)
        record_file = os.path.join(round_res_dir, "record.txt")

        with open(record_file, "w", encoding="utf-8") as record:
            for epoch in range(num_epochs):
                train(
                    model, device, train_loader, optimizer,
                    epoch + 1, graphs
                )

                # model/roundN/modelEPOCH/
                checkpoint_dir = os.path.join(
                    workdir,
                    "model",
                    f"round{round_index}",
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoint_dir, f"epoch_{epoch + 1}.pth"
                    ),
                )

                # Validation
                labels, predictions = predicting(
                    model, device, val_loader, graphs
                )
                valid_dir = os.path.join(
                    workdir, "result", f"round{round_index}", "valid"
                )
                valid_result_file = os.path.join(
                    valid_dir, f"result{epoch + 1}.txt"
                )
                save_result(labels, predictions, valid_result_file)

                valid_mse = mse(labels, predictions)
                valid_ci = get_cindex(labels, predictions)
                valid_rm2 = get_rm2(labels, predictions)

                if valid_mse < best_mse:
                    best_mse = valid_mse
                    best_epoch = epoch

                message = (
                    f"At epoch in validation dataset: {epoch + 1} "
                    f"MSE {valid_mse}; R2 {valid_rm2}; CI {valid_ci}"
                )
                print(message)
                record.write(message + "\n")

                # Test
                labels, predictions = predicting(
                    model, device, test_loader, graphs
                )
                test_dir = os.path.join(
                    workdir, "result", f"round{round_index}", "test"
                )
                test_result_file = os.path.join(
                    test_dir, f"result{epoch + 1}.txt"
                )
                save_result(labels, predictions, test_result_file)

                test_mse = mse(labels, predictions)
                test_ci = get_cindex(labels, predictions)
                test_rm2 = get_rm2(labels, predictions)

                message = (
                    f"At epoch in test dataset: {epoch + 1} "
                    f"MSE {test_mse}; R2 {test_rm2}; CI {test_ci}"
                )
                print(message)
                record.write(message + "\n\n")
                record.flush()

            record.write(f"Best epoch {best_epoch + 1}\n")
            record.write(f"Best validation MSE {best_mse}\n")

        print(
            f"Round {round_index} completed. "
            f"Best epoch: {best_epoch + 1}; best validation MSE: {best_mse}"
        )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python train_dta.py <workdir> <gpu_id>\n"
            "Example: python train_dta.py ./test_example/ 0"
        )

    workdir = os.path.abspath(sys.argv[1])
    gpu_id = sys.argv[2]

    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"Working directory not found: {workdir}")

    # CUDA_VISIBLE_DEVICES is set before importing torch and DGL above.
    for round_index in range(1, 6):
        main(workdir, round_index)

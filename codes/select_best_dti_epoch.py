import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, roc_auc_score, average_precision_score, roc_curve

def read_result(result_file):

    f = open(result_file, "r")
    text = f.read()
    f.close()

    pred = []
    real = []

    for line in text.splitlines():
        values = line.strip().split()
        real.append(float(values[0]))
        pred.append(float(values[1]))

    return np.array(pred), np.array(real)

def save_result(G, P, result_file):

    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()

def get_average_result(workdir, data_type, round_times, iteration_number):

    result_file = workdir + "/round1/" + data_type + "/result" + str(iteration_number) + ".txt"

    pred, real = read_result(result_file)

    for i in range(2, round_times + 1):

        result_file = workdir + "/round" + str(i) + "/" + data_type + "/result" + str(iteration_number) + ".txt"

        s_pred, _ = read_result(result_file)
        pred = pred + s_pred

    pred = pred/round_times

    return pred, real

def create_result(workdir, round_times, max_iteration):

    best_auroc = 0
    best_epoch = 0

    res = os.path.join(workdir, "final_record")
    print("=" * 72)
    print("DTI Best-Epoch Selection")
    print("=" * 72)
    print(f"Working directory : {os.path.abspath(workdir)}")
    print(f"Training rounds   : {round_times}")
    print(f"Epochs to evaluate: {max_iteration}")
    print("Selection metric  : Validation AUROC (maximum)")
    print("Aggregation       : Mean prediction across training rounds")
    print("-" * 72)

    f = open(res, "w", encoding="utf-8")

    for epoch in range(1, max_iteration + 1):

        P, G = get_average_result(workdir, "valid", round_times, epoch)
        auroc = roc_auc_score(G, P)

        if (auroc > best_auroc):
            best_epoch = epoch
            best_auroc = auroc

        precision, recall, _ = metrics.precision_recall_curve(G, P)
        auprc = metrics.auc(recall, precision)
        print(
            f"[Validation] Epoch {epoch:02d}/{max_iteration} | "
            f"AUROC: {auroc:.6f} | AUPRC: {auprc:.6f}"
        )
        f.write(f'At epoch: {epoch} AUROC {auroc} ; AUPRC {auprc}\n')

        result_dir = workdir + "/final_result/valid"
        if (os.path.exists(result_dir) == False):
            os.makedirs(result_dir)
        result_file = result_dir + "/result" + str(epoch) + ".txt"
        save_result(G, P, result_file)

        P, G = get_average_result(workdir, "test", round_times, epoch)

        result_dir = workdir + "/final_result/test"
        if (os.path.exists(result_dir) == False):
            os.makedirs(result_dir)
        result_file = result_dir + "/result" + str(epoch) + ".txt"
        save_result(G, P, result_file)

        auroc = roc_auc_score(G, P)

        precision, recall, thresholds = metrics.precision_recall_curve(G, P)

        auprc = metrics.auc(recall, precision)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # 防止除零错误

        thred_optim = thresholds[5:][np.argmax(f1[5:])]
        print(
            f"[Test] Optimal F1 threshold at epoch {epoch}: "
            f"{thred_optim:.6f}"
        )
        y_pred_s = [1 if i else 0 for i in (P >= thred_optim)]
        mcc = metrics.matthews_corrcoef(G, y_pred_s)
        F1 = metrics.f1_score(G, y_pred_s)

        cm1 = confusion_matrix(G, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        print(
            f"[Test]       Epoch {epoch:02d}/{max_iteration} | "
            f"AUROC: {auroc:.6f} | AUPRC: {auprc:.6f} | "
            f"Accuracy: {accuracy:.6f} | Sensitivity: {sensitivity:.6f} | "
            f"Specificity: {specificity:.6f} | MCC: {mcc:.6f} | F1: {F1:.6f}"
        )

        f.write(f'Test At epoch: {epoch}, AUROC {auroc} ; AUPRC {auprc} ; Accuracy {accuracy} ; Sensitivity {sensitivity} ; Specificity {specificity} ; MCC {mcc} ; F1 {F1}\n')
        f.write("\n")
        f.flush()

    f.write(
        f"\nBest epoch (selected by validation AUROC): {best_epoch}\n"
        f"Best validation AUROC: {best_auroc}\n"
    )

    f.close()

    print("-" * 72)
    print(f"Best epoch           : {best_epoch}")
    print(f"Best validation AUROC: {best_auroc:.6f}")
    print(f"Evaluation record    : {res}")
    print("=" * 72)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python 01_select_best_dti_epoch.py <workdir>\n"
            "Example: python 01_select_best_dti_epoch.py ./test_example/result/\n"
            "The working directory must contain round1/ through round5/, "
            "each with valid/ and test/ prediction files."
        )

    workdir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"Working directory not found: {workdir}")

    round_times = 5
    max_iteration = 50
    create_result(workdir, round_times, max_iteration)














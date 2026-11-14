import sys
import os
from scipy import stats
import numpy as np
from math import sqrt

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

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def get_cindex(y, p):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()

    y_diff = y[:, None] - y[None, :]
    p_diff = p[:, None] - p[None, :]

    mask = y_diff > 0

    if np.sum(mask) == 0:
        return np.nan

    score = (p_diff > 0).astype(float) + 0.5 * (p_diff == 0).astype(float)

    return np.sum(score[mask]) / np.sum(mask)

def r_squared_error(y_obs,y_pred):

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):

    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig,ys_line):

    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2-r02))))

def get_average_result(workdir, data_type, round_times, iteration_number):

    result_file = os.path.join(workdir, "round1", data_type, f"result{iteration_number}.txt")

    pred, real = read_result(result_file)

    for i in range(2, round_times + 1):

        result_file = os.path.join(workdir, f"round{i}", data_type, f"result{iteration_number}.txt")

        s_pred, _ = read_result(result_file)
        pred = pred + s_pred

    pred = pred/round_times

    return pred, real

def create_result(workdir, round_times, max_iteration):


    best_mse = 100000
    best_epoch = 0

    res = workdir + "/final_record"
    f = open(res, 'w')

    for epoch in range(1, max_iteration + 1):

        P, G = get_average_result(workdir, "valid", round_times, epoch)
        mse_loss = mse(G, P)
        concordance_index = get_cindex(G, P)
        rm2_value = get_rm2(G, P)

        if (mse_loss < best_mse):
            best_mse = mse_loss
            best_epoch = epoch

        print('At epoch in validation dataset: ', str(epoch), 'MSE ', str(mse_loss), '; R2 ', str(rm2_value),
              '; CI ' + str(concordance_index))
        f.write(
            f'At epoch in validation dataset: {str(epoch)} MSE {str(mse_loss)} ; R2 {str(rm2_value)}; CI {str(concordance_index)}\n')

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

        mse_loss = mse(G, P)
        concordance_index = get_cindex(G, P)
        rm2_value = get_rm2(G, P)

        print(f'[Test]       Epoch {epoch}/{max_iteration} | MSE: {mse_loss:.6f} | RM2: {rm2_value:.6f} | CI: {concordance_index:.6f}')
        print()
        f.write(f'At epoch in test dataset: {str(epoch)} MSE {str(mse_loss)} ; R2 {str(rm2_value)}; CI {str(concordance_index)}\n\n')

        f.flush()

    # Report and save the optimal epoch selected by validation MSE.
    best_message = (
        f"Best epoch (selected by validation MSE): {best_epoch}\n"
        f"Best validation MSE: {best_mse}"
    )
    print("=" * 65)
    print(best_message)
    print("=" * 65)

    f.write("\n" + "=" * 65 + "\n")
    f.write(best_message + "\n")
    f.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python evaluate_dta.py <workdir>\n"
            "Example: python evaluate_dta.py ./test_example/\n"
            "Expected predictions: <workdir>/res/roundN/{valid,test}/resultE.txt"
        )

    workdir = os.path.abspath(sys.argv[1])
    round_times = 5
    max_iteration = 100

    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"Working directory not found: {workdir}")

    print("=" * 65)
    print("DTA Model Evaluation")
    print("=" * 65)
    print(f"Working directory : {workdir}")
    print(f"Training rounds   : {round_times}")
    print(f"Epochs per round  : {max_iteration}")
    print("Prediction aggregation: mean across independent rounds")
    print("Best epoch selection: minimum validation MSE")
    print("-" * 65)

    # Check that all expected prediction files exist before evaluation.
    missing_files = []
    for round_index in range(1, round_times + 1):
        for data_type in ("valid", "test"):
            for epoch in range(1, max_iteration + 1):
                path = os.path.join(
                    workdir, f"round{round_index}",
                    data_type, f"result{epoch}.txt"
                )
                if not os.path.isfile(path):
                    missing_files.append(path)

    if missing_files:
        examples = "\n".join(f"  - {p}" for p in missing_files[:10])
        extra = len(missing_files) - min(10, len(missing_files))
        if extra:
            examples += f"\n  ... and {extra} more"
        raise FileNotFoundError(
            f"Missing {len(missing_files)} prediction file(s). "
            "Please finish DTA training first.\n" + examples
        )

    print("All prediction files found.")
    print("Starting ensemble evaluation...")
    print("-" * 65)

    create_result(workdir, round_times, max_iteration)

    print("-" * 65)
    print("DTA model evaluation completed.")
    print(f"Evaluation log : {os.path.join(workdir, 'final_record')}")
    print(f"Aggregated predictions : {os.path.join(workdir, 'final_result')}")
    print("=" * 65)

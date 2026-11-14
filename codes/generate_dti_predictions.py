import os
import sys

# Set GPU visibility before importing PyTorch/DGL.
if len(sys.argv) >= 3:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

import torch.nn as nn
from utils import *
from dti_model import *
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, roc_auc_score, \
    average_precision_score, roc_curve

from sklearn import metrics
import pickle as pkl
import dgl
from config import dataset_name, graph_name


def read_pkl(file_path):
    with open(file_path, 'rb') as fr:
        return pkl.load(fr)

TAU = 0.07
# The working directory contains dataset/, graph.pkl, model/, and result/.
# Loaded and validated in the CLI entry point below.
graphs = None


def predicting(model, device, loader, mode):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))

    with torch.no_grad():
        for data in loader:
            batch_indices = data.pro

            batch_graphs = [graphs[i.item()] for i in batch_indices]

            batch_graphs = dgl.batch(batch_graphs).to(device)
            feats = batch_graphs.ndata['x']

            data = data.to(device)
            output, z_t, z_d = model(data, batch_graphs, feats)
            if(mode==1):
                score = (z_d * z_t).sum(dim=-1)  # [B]
                output = torch.sigmoid(score/TAU)

            # output=model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


loss_fn = nn.BCELoss()


def save_result(G, P, result_file):
    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()


def main(workdir, round_index, mode, checkpoint_epoch):
    modeling = [QSI]
    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)
    LOG_INTERVAL = 200
    NUM_EPOCHS = 50
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    LR = 0.00003  ##0.00005
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # Main loop: iterate over different models
    for model in modeling:
        model_st = model.__name__
        print('\nrunning on ', model_st + '_' + os.path.basename(workdir))
        processed_data_file_train = os.path.join(workdir, dataset_name, 'train.pt')
        processed_data_file_val = os.path.join(workdir, dataset_name, 'val.pt')
        processed_data_file_test = os.path.join(workdir, dataset_name, 'test.pt')

        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (
        not os.path.isfile(processed_data_file_test))):
            raise FileNotFoundError('Preprocessed datasets are missing; run data_process.py first.')
        else:
            train_data = MyDataset(root=workdir, workdir=workdir, dataset=os.path.join(dataset_name, 'train.pt'))
            val_data = MyDataset(root=workdir, workdir=workdir, dataset=os.path.join(dataset_name, 'val.pt'))
            test_data = MyDataset(root=workdir, workdir=workdir, dataset=os.path.join(dataset_name, 'test.pt'))

            # loding the dataset
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = model().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.000001)


            record_dir = os.path.join(workdir, 'result', f'mode{mode}', f'round{round_index}')
            os.makedirs(record_dir, exist_ok=True)
            res = os.path.join(record_dir, 'record.txt')
            f = open(res, 'w')

            # ====== 改动点：不训练，直接加载 checkpoint ======
            ckpt_path = os.path.join(workdir, 'model', f'round{round_index}', f'epoch_{checkpoint_epoch}.pth')
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f'Model checkpoint not found: {ckpt_path}')
            print('Loading checkpoint:', ckpt_path)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)

            G, P = predicting(model, device, val_loader, mode)

            result_dir = os.path.join(record_dir, 'valid')
            if (os.path.exists(result_dir) == False):
                os.makedirs(result_dir)

            result_file = result_dir + "/result1.txt"
            save_result(G, P, result_file)

            auroc = roc_auc_score(G, P)
            precision, recall, _ = metrics.precision_recall_curve(G, P)
            auprc = metrics.auc(recall, precision)
            print('At validation dataset: ', 'AUROC', auroc, '; AUPRC', auprc)
            f.write(f'At  validation dataset: AUROC {auroc} ; AUPRC {auprc}\n')

            # torch.save(model.state_dict(), para_file_name)
            G, P = predicting(model, device, test_loader, mode)

            result_dir = os.path.join(record_dir, 'test')
            if (os.path.exists(result_dir) == False):
                os.makedirs(result_dir)
            result_file = result_dir + "/result1.txt"
            save_result(G, P, result_file)

            auroc = roc_auc_score(G, P)
            precision, recall, _ = metrics.precision_recall_curve(G, P)

            auprc = metrics.auc(recall, precision)

            fpr, tpr, thresholds = roc_curve(G, P)

            # print(fpr)
            # print(tpr)
            # print(thresholds)
            # print(thresholds.shape)
            precision, recall, thresholds = precision_recall_curve(G, P)

            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # 防止除零错误

            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            # print(f1)
            # print(f1.shape)
            # print(thred_optim)
            y_pred_s = [1 if i else 0 for i in (P >= thred_optim)]
            mcc = metrics.matthews_corrcoef(G, y_pred_s)
            F1 = metrics.f1_score(G, y_pred_s)

            cm1 = confusion_matrix(G, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            print('***')
            print('At test dataset: ', 'AUROC', auroc, '; AUPRC', auprc, '; Accuracy', accuracy,
                  '; Sensitivity', sensitivity, '; Specificity', specificity, '; MCC', mcc, '; F1', F1)
            print('***')

            f.write(f'At test dataset: AUROC {auroc} ; AUPRC {auprc} ; Accuracy {accuracy} ; Sensitivity {sensitivity} ; Specificity {specificity} ; MCC {mcc} ; F1 {F1}\n')
            f.write("\n")
            f.flush()


            f.close()


if __name__ == '__main__':

    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            'Usage: python load_dti_prediction.py <workdir> <gpu_id> [checkpoint_epoch]\n'
            'Example: python load_dti_prediction.py ./test_example/ 0 50\n'
            'Both Mode 0 (BCE prediction) and Mode 1 '
            '(contrastive similarity prediction) will be executed.'
        )

    workdir = os.path.abspath(sys.argv[1])
    gpu_id = sys.argv[2]
    checkpoint_epoch = int(sys.argv[3]) if len(sys.argv) == 4 else 50

    if checkpoint_epoch < 1:
        raise ValueError('checkpoint_epoch must be >= 1')

    if not os.path.isdir(workdir):
        raise NotADirectoryError(
            f'Working directory not found: {workdir}'
        )

    graph_file = os.path.join(workdir, graph_name + '.pkl')
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(
            f'Protein graph file not found: {graph_file}'
        )

    print('Loading protein graphs:', graph_file)
    graphs = read_pkl(graph_file)

    for mode in (0, 1):
        print('\n' + '=' * 60)
        print(f'Starting Mode {mode}: '
              f'{"BCE prediction" if mode == 0 else "Contrastive similarity prediction"}')
        print('=' * 60)

        for round_index in range(1, 6):
            main(workdir, round_index, mode, checkpoint_epoch)

    print('\nBoth Mode 0 and Mode 1 predictions completed successfully.')
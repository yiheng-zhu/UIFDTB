import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from utils import *
import sys
from config import dataset_name
from common_methods import create_dir

def main(workdir):

    compound_iso_smiles = []
    opts = ['train', 'val', 'test']
    for opt in opts:
        df = pd.read_csv(os.path.join(workdir, opt + ".csv"))
        compound_iso_smiles += list(df['SMILES'])

    compound_iso_smiles = set(compound_iso_smiles)

    smile_graph = {}
    for smile in tqdm(compound_iso_smiles):
        g = SMILE2GRAPH(smile)  # c_size, features, edge_index  all deliever to g
        smile_graph[smile] = g

        # convert to PyTorch data format, preparing for training the model
    processed_data_file_train = dataset_name + "/train.pt"
    processed_data_file_val = dataset_name + "/val.pt"
    processed_data_file_test = dataset_name + "/test.pt"

    create_dir(os.path.join(workdir, dataset_name))

    if (not os.path.isfile(os.path.join(workdir, processed_data_file_train)) or not os.path.isfile(os.path.join(workdir, processed_data_file_val)) or not os.path.isfile(os.path.join(workdir, processed_data_file_test))):
        df = pd.read_csv(os.path.join(workdir, 'train.csv'))
        train_drugs, train_prots, train_Y, train_protein_index, train_smile_index = list(df['SMILES']), list(
            df['Protein']), list(df['Y']), list(df['Protein_index']), list(df['SMILES_index'])
        train_drugs, train_prots, train_Y, train_protein_index, train_smile_index = np.asarray(
            train_drugs), np.asarray(train_prots), np.asarray(train_Y), np.asarray(train_protein_index), np.asarray(
            train_smile_index)

        df = pd.read_csv(os.path.join(workdir, 'val.csv'))
        val_drugs, val_prots, val_Y, val_protein_index, val_smile_index = list(df['SMILES']), list(
            df['Protein']), list(df['Y']), list(df['Protein_index']), list(df['SMILES_index'])
        val_drugs, val_prots, val_Y, val_protein_index, val_smile_index = np.asarray(val_drugs), np.asarray(
            val_prots), np.asarray(val_Y), np.asarray(val_protein_index), np.asarray(val_smile_index)

        df = pd.read_csv(os.path.join(workdir, 'test.csv'))
        test_drugs, test_prots, test_Y, test_protein_index, test_smile_index = list(df['SMILES']), list(
            df['Protein']), list(df['Y']), list(df['Protein_index']), list(df['SMILES_index'])
        test_drugs, test_prots, test_Y, test_protein_index, test_smile_index = np.asarray(test_drugs), np.asarray(
            test_prots), np.asarray(test_Y), np.asarray(test_protein_index), np.asarray(test_smile_index)

        print('preparing ', processed_data_file_train + ' in pytorch format!')
        train_data = MyDataset(root='./', workdir = workdir, dataset = processed_data_file_train, data_type = "train",
                               xd=train_drugs, xt=train_prots, xt_index=train_protein_index,
                               xd_index=train_smile_index, y=train_Y,
                               smile_graph=smile_graph)
        print('preparing ', processed_data_file_val + ' in pytorch format!')
        val_data = MyDataset(root='./', workdir = workdir, dataset=processed_data_file_val,  data_type = "val",
                             xd=val_drugs, xt=val_prots, xt_index=val_protein_index, xd_index=val_smile_index, y=val_Y,
                             smile_graph=smile_graph)
        print('preparing ', processed_data_file_test + ' in pytorch format!')
        test_data = MyDataset(root='./', workdir = workdir, dataset=processed_data_file_test, data_type = "test",
                              xd=test_drugs, xt=test_prots, xt_index=test_protein_index, xd_index=test_smile_index, y=test_Y,
                              smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_val, ' and ', processed_data_file_test,
              ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_val, ' and ', processed_data_file_test,
              ' are already created')

if __name__ == "__main__":

    main(sys.argv[1])

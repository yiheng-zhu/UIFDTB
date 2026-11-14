import os
import numpy as np
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import trange
import networkx as nx
from config import remove_record_file, chemberta_feature_dir, \
    interproscan_workspace, interproscan_featuredir, morgan_feature_dim, sequence_cut_off
import csv

class MyDataset(InMemoryDataset):
    def __init__(self, root='/tmp', workdir = "", dataset = "", data_type="",
                 xd=None, xt=None,xt_index=None, xd_index=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        # benchmark dataset, default = 'davis'

        self.workdir = os.path.abspath(workdir)
        self.data_type = data_type

        self.dataset_dir = os.path.dirname(dataset)
        self.dataset = os.path.basename(dataset)

        super(MyDataset, self).__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])   #_use_new_zipfile_serialization=False
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(workdir, data_type, xd, xt, xt_index, xd_index, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return os.path.join(self.workdir, self.dataset_dir)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, workdir, data_type, xd, xt, xt_index, xd_index, y, smile_graph):

        remove_file = os.path.join(workdir, data_type + "_" + remove_record_file)

        assert (len(xd) == len(xt) and len(xt) == len(y)), \
            "The three lists must be the same length!"

        data_list = []
        data_len = len(xd)

        with open(remove_file, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["SMILES_index", "Protein_index"])

            for i in trange(data_len):

                smiles = xd[i]
                target = xt[i].upper()
                target_index = xt_index[i]
                smile_index = xd_index[i]
                labels = y[i]

                c_size, features, edge_index = smile_graph[smiles]

                if len(edge_index) == 0:
                    writer.writerow([smile_index, target_index])
                    continue

                GCNData = DATA.Data(
                    x=torch.Tensor(features),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    y=torch.FloatTensor([labels])
                )

                GCNData.__setitem__('pro', torch.LongTensor([target_index]))

                GCNData.fp = torch.LongTensor([SMILE2ECFP(smiles)])
                GCNData.ch = torch.FloatTensor(
                    [read_smile_embeddings(workdir, smile_index)]
                )
                GCNData.interpro = torch.LongTensor(
                    [read_interpro_features(workdir, target_index)]
                )

                GCNData.target = torch.LongTensor([myseq_cat(target)])
                GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

                data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
# functions needed

from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')

#处理SMILE
def SMILE2ECFP(smile):
    mol = Chem.MolFromSmiles(smile)
    # 计算分子的Morgan指纹
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_feature_dim).ToBitString()
    # 1024维01数组
    fp_list = list(map(int,list(fp)))
    #print(fp_list)
    return fp_list

def read_smile_embeddings(workdir, smile_index):

    smile_embedding = np.load(os.path.join(workdir, chemberta_feature_dir, "mean_" + str(smile_index) + ".npy"))

    return smile_embedding.tolist()

def read_interpro_features(workdir, target_index):

    interpro_feature = np.load(os.path.join(workdir, interproscan_workspace, interproscan_featuredir, str(target_index) + ".npy"))

    return interpro_feature.tolist()


def myseq_cat(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    x = np.zeros(sequence_cut_off)
    for i, ch in enumerate(prot[:sequence_cut_off]):
        x[i] = seq_dict[ch]
    return x


def SMILE2GRAPH(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size=mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
    
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
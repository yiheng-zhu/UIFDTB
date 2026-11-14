import sys

import numpy as np
from tqdm.auto import trange
import math
import pickle as pkl
import dgl
import torch
import os
from config import graph_name, prottrans_featuredir, point_name, esmfold_structure_dir, distance_cutoff

def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()

def get_dis(point1, point2):
    dis_x = point1[0] - point2[0]
    dis_y = point1[1] - point2[1]
    dis_z = point1[2] - point2[2]
    return math.sqrt(dis_x*dis_x + dis_y*dis_y + dis_z*dis_z)


def get_whole_pdb_graph(workdir, pdb_points, pid_list, thresholds):
    pdb_graphs = []

    for pid in trange(pid_list):
        points = pdb_points[pid]

        u_list = []
        v_list = []
        dis_list = []
        for uid, amino_1 in enumerate(points):
            for vid, amino_2 in enumerate(points):
                if uid==vid:
                    u_list.append(uid)
                    v_list.append(vid)
                    dis_list.append(0)
                    continue

                dist = get_dis(amino_1, amino_2)
                if dist<=thresholds:
                    u_list.append(uid)
                    v_list.append(vid)
                    dis_list.append(dist)
        u_list, v_list = torch.tensor(u_list), torch.tensor(v_list)
        dis_list = torch.tensor(dis_list)

        graph = dgl.graph((u_list, v_list), num_nodes=len(points))
        graph.edata['dis'] = dis_list

        # graph node feature
        esm_features = torch.load(os.path.join(workdir, prottrans_featuredir, str(pid) + ".pt"))
        node_features = esm_features
        assert node_features.shape[0]==graph.num_nodes()
        graph.ndata['x'] = torch.from_numpy(node_features)

        pdb_graphs.append(graph)


    save_pkl(os.path.join(workdir, graph_name + ".pkl"), pdb_graphs)


def main(workdir):

    pdb_points=read_pkl(os.path.join(workdir, point_name + ".pkl"))
    nums=len(os.listdir(os.path.join(workdir, esmfold_structure_dir)))
    get_whole_pdb_graph(workdir, pdb_points, nums, distance_cutoff)


if __name__ == '__main__':

    main(sys.argv[1])
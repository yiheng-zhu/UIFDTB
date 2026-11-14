import os
import pickle as pkl
import sys
import pandas as pd
from tqdm import trange
from config import esmfold_structure_dir, protein_csv_file, point_name

def read_pdb(pdb_file, protein_length):
    #construct pdb ca points
    points = []
    with open(pdb_file, 'r') as f:
        for line in f:
            
            if line.startswith('ATOM'):
                point_type = line[12:16].strip() #col 13-16
                if point_type == 'CA':
                    x = float(line[30:38].strip()) #col 31-38
                    y = float(line[38:46].strip()) #col 39-46
                    z = float(line[46:54].strip()) #col 47-54
                    amino = line[17:20].strip() #col 18-20
                    points.append((x, y, z, amino))
    if len(points)>0:
        try:
            assert protein_length==len(points)
        except:
            return False
    return points


def main(workdir):

    file=len(os.listdir(os.path.join(workdir, esmfold_structure_dir)))
    protein_list=pd.read_csv(os.path.join(workdir, protein_csv_file))["Protein"].tolist()
    protein_length = []
    
    for i in range(len(protein_list)):

        protein = protein_list[i]
        protein_length.append(len(protein))

    pdb_points_info = {}
    for j in trange(file):
        pdb_file = os.path.join(workdir, esmfold_structure_dir, str(j) + ".pdb")
        if os.path.exists(pdb_file):
            acid_points = read_pdb(pdb_file, protein_length[j])
            if acid_points==False:
                print("Wrong sequence length!!!")
                #pdb_points_info[j] = acid_points
                return False
            elif len(acid_points)==0:
                print("Empty PDB file!!!")
                return False
            else:
                pdb_points_info[j] = acid_points
        else:
            print("Unseen proteins!!!")
            return False
    with open(os.path.join(workdir, point_name + ".pkl"), 'wb') as fw:
        pkl.dump(pdb_points_info, fw)
        print("Result save as: " + os.path.join(workdir, point_name + ".pkl"))


if __name__ == '__main__':

    main(sys.argv[1])
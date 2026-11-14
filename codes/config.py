import os

#protein
protein_csv_file = "protein.csv"
protein_fasta_file = "protein.fasta"
sequence_cut_off = 1000


#smiles
smiles_csv_file = "smile.csv"
max_smile_length = 500

remove_record_file = "remove_target_drug_pairs.csv"
dataset_name = "dataset"


##########################################################################################################################

# gpu
device_type = "gpu"
device_type_dict = dict()
device_type_dict["gpu"] = dict()
device_type_dict["gpu"]["device"] = "cuda:0"
device_type_dict["gpu"]["device_id"] = "0"
device_type_dict["gpu"]["gpu_ratio"] = 1.0
device_type_dict["cpu"] = dict()
device_type_dict["cpu"]["device"] = "cpu"
device_type_dict["cpu"]["device_id"] = ""

##########################################################################################################################




##########################################################################################################################

#interproscan
interproscan_workdir = "/data/yihengzhu/toolbars/sequence_homology_tools/InterPro/interproscan-5.69-101.0/"
interproscan_script_file = os.path.join(interproscan_workdir, "interproscan.sh")
interproscan_entry_list_file = os.path.join(interproscan_workdir, "entry.list")
interproscan_workspace = "interpro"
interproscan_result_filename = "inital_interpro_result"
interproscan_temp_filedir = "interpro_temp"
interproscan_resultdir = "interpro_results"
interproscan_featuredir = "interpro_array"
interpro_feature_dim = 45899

##########################################################################################################################


##########################################################################################################################

#prottrans
prottrans_featuredir = "prottrans_features"
prottrans_feature_dim = 1024

##########################################################################################################################


##########################################################################################################################

#esmfold
esmfold_structure_dir = "esmfold"
max_gpu_length = 600
point_name = "pdb_points"
graph_name = "graph"
distance_cutoff = 10

##########################################################################################################################

##########################################################################################################################

#drug feature representation
chemberta_feature_dir = "chemberta_features"
chemberta_feature_dim = 768
morgan_feature_dim = 1024

##########################################################################################################################
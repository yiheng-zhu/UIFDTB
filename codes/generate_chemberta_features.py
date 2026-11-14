import os.path
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import trange
from rdkit import Chem
import numpy as np
from config import max_smile_length, device_type_dict, device_type, smiles_csv_file, chemberta_feature_dir
from common_methods import create_dir


def count_molecules_rdkit(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumAtoms()


os.environ["CUDA_VISIBLE_DEVICES"] = device_type_dict[device_type]["device_id"]
device = torch.device(device_type_dict[device_type]["device"])

MODEL = "DeepChem/ChemBERTa-100M-MLM"


def main(workdir):

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    model.eval()

    smiles_list = pd.read_csv(os.path.join(workdir, smiles_csv_file))['SMILES'].to_list()

    feature_dir = os.path.join(workdir, chemberta_feature_dir)
    create_dir(feature_dir)

    for i in trange(len(smiles_list)):
        smiles = smiles_list[i]
        atom_num = count_molecules_rdkit(smiles)

        inputs = tokenizer(
            smiles,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_smile_length
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)


        # mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb_mean = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb_mean = emb_mean[0]
        feature_file = os.path.join(feature_dir, "mean_" + str(i) + ".npy")
        np.save(feature_file, emb_mean.detach().cpu().numpy())


if __name__ == "__main__":

    print("Generate ChmeBERTa embeddings:")
    main(sys.argv[1])





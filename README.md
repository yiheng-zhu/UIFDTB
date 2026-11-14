# UIFDTB

## A Unified and Interpretable Framework for Drug–Target Binding Prediction

## 1. Introduction

<div align="justify">

UIFDTB is a unified and interpretable deep learning framework for drug–target binding prediction, supporting both drug–target interaction (DTI) classification and drug–target affinity (DTA) regression.

</div>

## 2. Framework Overview

<div align="justify">

The overall architecture of UIFDTB, illustrated below, comprises four main components:

</div>

<div align="justify">

<ul>
<li><strong>Target Encoding:</strong> Extracts complementary functional-domain and structure-aware target representations.</li>

<li><strong>Drug Encoding:</strong> Extracts multi-view drug representations from Morgan fingerprints, ChemBERTa embeddings, and molecular graphs.</li>

<li><strong>Interaction Modeling:</strong> Integrates global drug and target representations through learnable gating and captures fine-grained atom–residue interactions using bidirectional cross-attention.</li>

<li><strong>Task-specific Prediction:</strong> Refines the joint drug–target representation through feature-wise attention and generates DTI interaction probabilities or DTA binding affinity scores.</li>
</ul>

</div>

![Overview of UIFDTB](./Figure1.tif)

## 3. Requirements

<div align="justify">

The current implementation was developed and tested with the following environment:

</div>

- Python: 3.8.19
- PyTorch: 2.0.0
- CUDA: 11.8
- cuDNN: 8.7.0
- pandas: 2.0.3
- transformers: 4.46.3
- numpy: 1.23.0
- dgl: 1.1.0+cu118
- rdkit: 2024.09.4
- torch_geometric: 2.6.1

## 4. Dataset Preparation

<div align="justify">

The benchmark datasets used in the UIFDTB paper are provided in the `dataset/` directory. These datasets cover both DTI classification and DTA regression tasks under different evaluation settings.

</div>

<div align="justify">

The provided datasets include:

</div>

| Dataset | Evaluation Setting |
|---|---|
| BIOSNAP | Random split |
| BIOSNAP | Drug-cold |
| BIOSNAP | Target-cold |
| BindingDB | Random split |
| Human | Random split |
| Human | Pair-cold |
| Human | Drug-cold |
| Davis | Random split |
| KIBA | Random split |

<div align="justify">

Each dataset directory contains five CSV files:

</div>

- `train.csv`: Training drug–target pairs and their labels.
- `val.csv`: Validation drug–target pairs and their labels.
- `test.csv`: Test drug–target pairs and their labels.
- `protein.csv`: Protein sequences indexed by `Protein_index`.
- `smile.csv`: Drug SMILES strings indexed by `SMILES_index`.

<div align="justify">

The dataset split files contain the following columns:

</div>

- `Y`: Binary interaction label (1 for interacting pairs and 0 for non-interacting pairs) or continuous affinity values for DTA prediction.
- `SMILES_index`: Index of the corresponding drug molecule.
- `Protein_index`: Index of the corresponding target protein.

<div align="justify">

The `SMILES_index` and `Protein_index` columns are used to retrieve the corresponding molecular and protein information from `smile.csv` and `protein.csv`, respectively.

</div>

<div align="justify">

**Example directory structure:**

</div>

```text
dataset/
├── BIOSNAP_drug_cold/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── protein.csv
│   └── smile.csv
├── BIOSNAP_random_split/
├── BIOSNAP_target_cold/
├── BindingDB/
├── Davis/
├── Human_pair_cold/
├── Human_random_split/
└── KIBA/
```
<div align="justify">

**Note:** Before running the code, ensure that the working directory contains all five required CSV files: `train.csv`, `val.csv`, `test.csv`, `protein.csv`, and `smile.csv`.

**Important:** The `SMILES_index` in `smile.csv` and the `Protein_index` in `protein.csv` must be consecutively numbered starting from 0 (i.e., 0, 1, 2, ..., N-1).

</div>

## 5. Code Implementation

### 5.1 Feature Representation
#### 5.1.1 Target Encoder

**Step 1: Preprocess protein sequences**

<div align="justify">

Before target feature extraction, preprocess the protein sequences in `protein.csv` using the following command:

</div>

```bash
python preprocess_protein.py ./test_example/
```

<div align="justify">

Here, `test_example` is the working directory containing `protein.csv`. The script automatically backs up the original file as `protein_original.csv` and performs the following preprocessing operations:

- Convert protein sequences to uppercase.
- Remove the non-standard amino acid symbols `B`, `J`, `O`, `U`, `X`, and `Z`.
- Truncate sequences longer than 1,000 amino acid residues.

The processed sequences are saved in `protein.csv`, replacing the original file.

</div>

<div align="justify">
    
**Step 2: Install InterProScan**

</div>

```bash
wget https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.69-101.0/interproscan-5.69-101.0-64-bit.tar.gz

tar -pxvzf interproscan-5.69-101.0-64-bit.tar.gz
```

<div align="justify">

**Step 3: Configure the InterProScan installation path**

</div>

<div align="justify">

Open `config.py` and modify the `interproscan_workdir` variable to specify the absolute path of the extracted InterProScan directory.

</div>

```python
interproscan_workdir = "/path/to/interproscan-5.69-101.0/"
```

<div align="justify">

**Step 4: Generate family-domain-based binary vectors**

</div>

<div align="justify">

Prepare `protein.csv` in the working directory. The file contains two columns: `Protein_index` and `Protein`, representing protein identifiers and amino acid sequences, respectively.

</div>

<div align="justify">

Run the following command to generate functional-domain-based binary vectors:

</div>

```bash
python generate_interpro_feature.py ./test_example/
```

<div align="justify">

Here, `test_example` is the working directory containing `protein.csv`. The protein sequences are processed by InterProScan to identify functional-domain annotations.

</div>

<div align="justify">

**Output directory structure:**

</div>

```text
test_example/
├── protein.csv
├── protein.fasta
└── interpro/
    ├── interpro_results/
    └── interpro_array/
        ├── 0.npy
        ├── 1.npy
        ├── 2.npy
        └── ...
```

<div align="justify">

<ul>
<li><code>interpro_results/</code>: Stores the original functional-domain annotation results generated by InterProScan.</li>

<li><code>interpro_array/</code>: Stores the generated Family-domain-based binary vectors in <code>.npy</code> format. Each protein is represented by a 45,899-dimensional binary vector, where each dimension indicates the presence or absence of a specific InterPro entry.</li>
</ul>

</div>


<div align="justify">

**Step 5: Generate ProtT5 embeddings**

<div align="justify">

Install ProtT5 and configure the required dependencies by following the instructions provided in the official [ProtTrans repository](https://github.com/agemagician/ProtTrans).

</div>

<div align="justify">

Run the following command to generate residue-level protein representations using the pre-trained ProtT5 model:

</div>

```bash
python generate_prottrans_feature.py ./test_example/
```

<div align="justify">

Here, `test_example` is the working directory containing `protein.csv`. Step 4 generates functional-domain annotations and automatically converts `protein.csv` into `protein.fasta`, which serves as the input for ProtT5 embedding extraction in Step 5. The resulting residue-level embeddings are saved in the `prottrans_features` directory, with a 1,024-dimensional vector for each amino acid residue.

</div>

**Output directory structure:**

```text
test_example/
├── protein.csv
├── protein.fasta
└── prottrans_features/
    ├── 0.pt
    ├── 1.pt
    ├── 2.pt
    └── ...
```

**Step 6: Predict protein structures using ESMFold**

<div align="justify">

Install ESMFold and its required dependencies by following the instructions in the official [ESM repository](https://github.com/facebookresearch/esm). Ensure that PyTorch and a compatible CUDA environment are installed before proceeding. ESMFold also requires OpenFold and its associated dependencies.

</div>

<div align="justify">

Run the following command to predict protein structures using the pre-trained ESMFold model:

</div>

```bash
python predict_structures_esmfold.py ./test_example/
```

<div align="justify">

Here, `test_example` is the working directory containing `protein.fasta`, which is generated during Step 4. The predicted protein structures are saved in the `esmfold` directory in PDB format (`.pdb`), with filenames corresponding to the protein identifiers in the FASTA file. Existing non-empty PDB files are automatically skipped.

</div>

**Output directory structure:**

```text
test_example/
├── protein.csv
├── protein.fasta
└── esmfold/
    ├── 0.pdb
    ├── 1.pdb
    ├── 2.pdb
    └── ...
```
**Step 7: Generate Protein Graphs**

<div align="justify">

Run the following commands to extract Cα atomic coordinates from ESMFold-predicted structures and construct protein graphs:

</div>

```bash
python generate_points.py ./test_example/
python generate_graphs.py ./test_example/
```

<div align="justify">

Here, `generate_points.py` extracts the three-dimensional coordinates and amino acid identities of Cα atoms from the predicted PDB files and saves them in `points.pkl`. Subsequently, `generate_graphs.py` integrates the extracted structural information with ProtT5 residue-level embeddings to construct protein graphs. Each amino acid residue is represented as a graph node, with its ProtT5 embedding as the node feature. Edges are established between residues whose Cα distances are within the predefined cutoff (`distance_cutoff` in `config.py`), with spatial distances stored as edge features. Self-loops are also included. The resulting protein graphs are saved in `graphs.pkl` in DGL format.

</div>

**Output directory structure:**

```text
test_example/
├── prottrans/
├── esmfold/
│   ├── 0.pdb
│   ├── 1.pdb
│   └── ...
├── points.pkl
└── graphs.pkl
```


#### 5.1.2 Drug Encoder
**Step 1: Generate ChemBERTa embeddings**

<div align="justify">

First, install the ChemBERTa environment by following the instructions in the official [ChemBERTa-3 repository](https://github.com/deepforestsci/chemberta3#setup-environment).

After installing the required dependencies, return to the UFIDTB project directory and run the following command to extract molecular representations using the pre-trained ChemBERTa model (`DeepChem/ChemBERTa-100M-MLM`):

</div>

```bash
python generate_chemberta_features.py ./test_example/
```

<div align="justify">

Here, `test_example` is the working directory containing `smile.csv`, which stores molecular SMILES strings in the `SMILES` column. The script applies mean pooling to the ChemBERTa token embeddings to obtain a 768-dimensional representation for each molecule. The resulting embeddings are saved as `.npy` files in the `chemberta` directory, with filenames corresponding to their molecule indices.

</div>

**Output directory structure:**

```text
test_example/
├── smile.csv
└── chemberta_features/
    ├── mean_0.npy
    ├── mean_1.npy
    ├── mean_2.npy
    └── ...
```

**Step 2: Construct molecular graphs and prepare PyTorch datasets**

<div align="justify">

Run the following command to construct molecular graphs and prepare the PyTorch datasets for training, validation, and testing:

</div>

```bash
python data_process.py ./test_example/
```

<div align="justify">

Here, `test_example` is the working directory containing the five input CSV files and the extracted drug and protein representations. The script performs the following operations:

- **Molecular graph construction:** Constructs molecular graphs from SMILES, representing atoms as nodes and chemical bonds as edges.
- **Morgan fingerprint extraction:** Generates 1,024-dimensional Morgan fingerprints for each molecule.
- **Feature integration:** Integrates Morgan fingerprints, ChemBERTa embeddings, InterProScan features, and protein sequence representations.
- **Dataset preparation:** Packages the resulting drug–target representations into PyTorch datasets for training, validation, and testing.

<div align="justify">

Drug–target pairs for which molecular graphs cannot be constructed are removed from the corresponding datasets and recorded in separate CSV files (e.g., `train_remove_target_drug_pairs.csv`), containing `SMILES_index` and `Protein_index`.

</div>

</div>

**Output directory structure:**

```text
test_example/
├── dataset/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
├── train_remove_target_drug_pairs.csv
├── val_remove_target_drug_pairs.csv
└── test_remove_target_drug_pairs.csv
```

### 5.2 Model Training

**Step 1: Train the DTA Model**

<div align="justify">

After completing the data preparation steps, run the following command to train the DTA prediction model:

</div>

```bash
python train_dta.py ./test_example/ 0
```

<div align="justify">

Here, `test_example` is the working directory containing the preprocessed PyTorch datasets in the `dataset` directory and the protein graphs in `graphs.pkl`. The second argument specifies the GPU index used for training (e.g., `0` for GPU 0).

The model is trained for 5 independent rounds, with 100 epochs per round, a batch size of 64, and a learning rate of 0.00003. During training, model performance is evaluated on both validation and test sets at each epoch using MSE, R², and concordance index (CI). The model checkpoints, prediction results, and training records are saved in the `model` and `result` directories, respectively.

</div>

**Output directory structure:**

```text
test_example/
│
├── model/
│   ├── round1/
│   │   ├── epoch_1.pth
│   │   ├── epoch_2.pth
│   │   ├── epoch_3.pth
│   │   ├── ...
│   │   └── epoch_100.pth
│   │
│   ├── ...
│   │
│   └── round5/
│       └── ...
│
└── result/
    ├── round1/
    │   ├── record.txt
    │   ├── valid/
    │   │   ├── result1.txt
    │   │   ├── result2.txt
    │   │   ├── ...
    │   │   └── result100.txt
    │   │
    │   └── test/
    │       ├── result1.txt
    │       ├── result2.txt
    │       ├── ...
    │       └── result100.txt
    │
    ├── ...
    │
    └── round5/
        └── ...
```

**Step 2: Train the DTI Model**

<div align="justify">

Run the following command to train the DTI prediction model:

</div>

```bash
python train_dti.py ./test_example/ 0
```

<div align="justify">

The model is trained for five independent rounds, with 100 epochs per round. A bidirectional contrastive loss is incorporated into the binary classification objective to enhance drug–target representation learning. Model checkpoints, training records, and prediction results are saved in the `model` and `result` directories, respectively.

</div>

### 5.3 Performance Evaluation

#### 5.3.1 Evaluate DTA Model Performance

<div align="justify">

After completing the five independent training rounds, run the following command to evaluate DTA model performance:

</div>

```bash
python evaluate_dta.py ./test_example/result/
```

<div align="justify">

Here, `test_example/result` is the working directory containing the prediction results from five independent training rounds. For each epoch, predictions from the five rounds are averaged to obtain ensemble predictions on the validation and test sets. Model performance is evaluated using mean squared error (MSE), the coefficient of determination (RM²), and the concordance index (CI).

The optimal epoch is selected based on the lowest validation MSE across all 100 epochs. The optimal epoch and its corresponding validation MSE are printed to the console and saved in `final_record`. Ensemble predictions for all epochs on the validation and test sets are saved in the `final_result` directory.

</div>

**Output directory structure:**

```text
test_example/
└── result/
    ├── final_record
    │
    └── final_result/
        ├── valid/
        │   ├── result1.txt
        │   ├── result2.txt
        │   ├── ...
        │   └── result100.txt
        │
        └── test/
            ├── result1.txt
            ├── result2.txt
            ├── ...
            └── result100.txt
```

<div align="justify">

The `final_record` file stores the evaluation metrics for each epoch and the optimal epoch selected based on validation MSE. The `final_result` directory contains the ensemble predictions, with each result file storing the ground-truth values and corresponding predicted values.

</div>

#### 5.3.2 Evaluate DTI Model Performance

<div align="justify">

DTI model evaluation consists of three steps. First, the optimal epoch is selected based on the average predictions from five independent training rounds. Next, the selected checkpoint is used to generate predictions from two complementary branches: BCE-based prediction (Mode 0) and contrastive similarity-based prediction (Mode 1). Finally, the predictions from the two branches are combined through weighted fusion to obtain the final DTI predictions.

</div>

**Step 1: Select the Optimal Epoch**

```bash
python 01_select_best_dti_epoch.py ./test_example/result/
```

<div align="justify">

The predictions from five independent training rounds are averaged at each epoch. The optimal epoch is selected based on the highest AUROC on the validation set across all 50 epochs. The evaluation metrics and optimal epoch are saved in `final_record`.

</div>

**Step 2: Generate DTI Predictions**

```bash
python 02_generate_dti_predictions.py ./test_example/ 0
```

<div align="justify">

The script automatically loads the optimal checkpoint identified in Step 1 and generates predictions using both branches. Mode 0 uses the BCE-based prediction branch, while Mode 1 uses the contrastive similarity-based prediction branch. Predictions are generated for all five training rounds and saved separately under the corresponding mode directories.

</div>

**Step 3: Search for the Optimal Fusion Weight**

```bash
python 03_search_dti_fusion_weight.py ./test_example/result/
```

<div align="justify">

The predictions from the two branches are combined using a weighted fusion strategy:

</div>

\[
\hat{y}=w\hat{y}_{0}+(1-w)\hat{y}_{1}
\]

<div align="justify">

where \(\hat{y}_{0}\) and \(\hat{y}_{1}\) denote the predictions from Mode 0 and Mode 1, respectively, and \(w\) is the fusion weight.

The fusion weight is searched from 0.01 to 1.00, with a step size of 0.01. The optimal weight is selected based on the highest AUROC on the validation set. The corresponding fused predictions are then evaluated on the test set using AUROC, AUPRC, Accuracy, Sensitivity, Specificity, MCC, and F1-score.

</div>

**Output directory structure:**

```text
test_example/
└── result/
    ├── final_record
    │
    ├── mode0/
    │   ├── round1/
    │   ├── ...
    │   └── round5/
    │
    ├── mode1/
    │   ├── round1/
    │   ├── ...
    │   └── round5/
    │
    └── combine_result/
        ├── valid/
        │   └── result.txt
        │
        └── test/
            └── result.txt
```

<div align="justify">

The `final_record` file stores the evaluation results for each epoch and the selected optimal epoch. The `mode0` and `mode1` directories contain predictions generated by the two prediction branches. The `combine_result` directory stores the final fused predictions for the validation and test sets.

</div>

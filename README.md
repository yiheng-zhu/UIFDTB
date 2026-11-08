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

</div>

## 5. Code Implementation

### 5.1 Feature Representation
#### 5.1.1 Target Encoder

**Step 1: Preprocess protein sequences**

<div align="justify">

Before feature extraction and ESMFold prediction, preprocess the protein sequences in `protein.csv` using the following command:

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

After execution, an `interpro` directory will be generated:

</div>

```text
test_example/
├── protein.csv
└── interpro/
    ├── interpro_results/
    └── interpro_array/
```

<div align="justify">

<ul>
<li><code>interpro_results/</code>: Stores the original functional-domain annotation results generated by InterProScan.</li>

<li><code>interpro_array/</code>: Stores the generated Family-domain-based binary vectors in <code>.npy</code> format. Each protein is represented by a 45,899-dimensional binary vector, where each dimension indicates the presence or absence of a specific InterPro entry.</li>
</ul>

</div>


<div align="justify">

**Step 5: Generate ProtT5 Embeddings**

</div>

<div align="justify">

Run the following command to generate residue-level protein representations using the pre-trained ProtT5 model:

</div>

```bash
python generate_prottrans_feature.py ./test_example/
```
<div align="justify">

Here, `test_example` is the working directory containing `protein.csv`. Step 3 generates functional-domain annotations and automatically converts `protein.csv` into `protein.fasta`, which serves as the input for ProtT5 embedding extraction in Step 4. The resulting residue-level embeddings are saved in the `prottrans` directory, with a 1,024-dimensional vector for each amino acid residue.

</div>

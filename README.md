# UFIDTB

## A Unified and Interpretable Framework for Drug–Target Binding Prediction

## 1. Introduction

UFIDTB is a unified and interpretable deep learning framework for drug–target binding prediction, supporting both drug–target interaction (DTI) classification and drug–target affinity (DTA) regression.

## 2. Framework Overview

The overall architecture of UFIDTB, illustrated below, comprises four main components:

- **Target Encoding:** Extracts complementary functional-domain and structure-aware target representations.
- **Drug Encoding:** Extracts multi-view drug representations from Morgan fingerprints, ChemBERTa embeddings, and molecular graphs.
- **Interaction Modeling:** Integrates global drug and target representations through learnable gating and captures fine-grained atom–residue interactions using bidirectional cross-attention.
- **Task-specific Prediction:** Refines the joint drug–target representation through feature-wise attention and generates DTI interaction probabilities or DTA binding affinity scores.

![Overview of UFIDTB](./Figure1.tif)

## 3. Requirements

The current implementation was developed and tested with the following environment:

- Python: 3.8.19
- PyTorch: 2.0.0
- CUDA: 11.8
- cuDNN: 8.7.0

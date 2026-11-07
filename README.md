# UFIDTB

## A Unified and Interpretable Framework for Drug–Target Binding Prediction

## Introduction

UFIDTB is a unified and interpretable deep learning framework for drug–target binding prediction, supporting both drug–target interaction (DTI) classification and drug–target affinity (DTA) regression.

## Framework Overview

The overall architecture of UFIDTB, illustrated below, comprises four main components:

- **Target Encoding:** Extracts complementary functional-domain and structure-aware representations.
- **Drug Encoding:** Extracts multi-view molecular representations from Morgan fingerprints, ChemBERTa embeddings, and molecular graphs.
- **Interaction Modeling:** Integrates global drug and target representations through learnable gating and captures fine-grained atom–residue interactions using bidirectional cross-attention.
- **Task-specific Prediction:** Refines the joint drug–target representation through feature-wise attention and generates DTI interaction probabilities or DTA binding affinity scores.

![Overview of UFIDTB](./Figure1.tif)


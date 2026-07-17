# DAPC-Net: A Density-Adaptive and Physics-Consistent Graph Neural Network for High-Entropy Alloy Screening

This repository contains the official PyTorch implementation of **DAPC-Net**. 

DAPC-Net is designed for the high-throughput screening of high-entropy alloy (HEA) electrocatalysts. By introducing a density-adaptive module and rigorous energy-force consistency constraints, our model achieves state-of-the-art (SOTA) accuracy in predicting potential energy surfaces (PES) and conservative force fields for complex atomistic systems.

This codebase is built upon the foundational [EquiformerV2](https://github.com/yilunliao/EquiformerV2) framework.

---

## 💡 Core Novelties & Code Implementation

To assist reviewers and researchers in easily locating our core theoretical contributions within the codebase, we have marked our primary modifications with `# idea1` and `# idea2` tags. 

You can globally search for these tags or directly inspect the following modified files:

### 1. Idea 1: Density-Adaptive Module (`# idea1`)
Designed to capture severe local lattice distortions typically found in high-entropy alloys.
* `nets/equiformer_v2/input_block.py`: Modified to adaptively process local density features.
* `nets/equiformer_v2/equiformer_v2_oc20.py`: Integrated the density-adaptive representations into the main model pipeline.

### 2. Idea 2: Energy-Force Consistency Constraint (`# idea2`)
Ensures that the predicted force fields are strictly conservative, a vital property for reliable molecular dynamics and catalyst screening.
* `oc20/trainer/forces_trainer_v2.py`: Modified the loss functions and training loops to enforce rigorous energy-force consistency during backpropagation.

### 3. Hyperparameters & Configuration
* `oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml`: Configured specific hyperparameters optimized for our curated OC20-derived HEA dataset.

---

## ⚙️ Environment Setup & Data Preparation

Since DAPC-Net is developed based on the EquiformerV2 and Open Catalyst Project (OCP) ecosystems, the environment setup is identical to the original framework.

1. **Environment:** Please refer to the [original setup guide](docs/env_setup.md) to install PyTorch, PyG, and related dependencies.
2. **Datasets:** DAPC-Net was evaluated on a curated OC20-derived HER dataset. To download the baseline OC20 datasets, please follow the [OCP GitHub repository instructions](https://github.com/Open-Catalyst-Project/ocp).

---

## 🚀 Training DAPC-Net

To train DAPC-Net using our modified configuration, run the following script (adjusting the number of GPUs/nodes as per your hardware):

```bash
sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@2M_g@8.sh
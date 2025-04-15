# Lithology Prediction Using ILCH-Net, LSTM, and AE Models

This repository contains code for predicting missing lithological data from well-log sequences using three different deep learning models: **ILCH-Net**, **LSTM**, and **Autoencoder (AE)**. The dataset used in this project is the **Volve dataset**, which is available on Equinor's site.

## Models

The following models are implemented in this repository:

### 1. **ILCH-Net (CVAE + LSTM)**

ILCH-Net is a deep-learning model that combines **Conditional Variational Autoencoders (CVAE)** with **Long Short-Term Memory (LSTM)** networks to predict missing well-log data. It refines predictions iteratively using geological constraints, achieving high accuracy even with incomplete data.

- Code: `Model_ILCH-Net.py`

### 2. **LSTM (Long Short-Term Memory)**

The LSTM model predicts missing data in well-log sequences by leveraging **Long Short-Term Memory (LSTM)** networks, which capture temporal dependencies and are well-suited for sequential data.

- Code: `Model_LSTM.py`

### 3. **Autoencoder (AE)**

The Autoencoder (AE) model learns a compressed representation of well-log sequences and predicts missing data based on this latent space representation.

- Code: `Model_AE.py`

### 4. **Iterative Autoencoder (Iter-AE)**

The Iterative Autoencoder (Iter-AE) enhances the standard AE model by applying an iterative refinement process to improve the reconstruction of missing well-log data.

- Code: `Model_Iter-AE.py`

## Data

The Volve dataset used in this study is available via the Equinor website:  
[https://www.equinor.com/news/archive/14jun2018-disclosing-volve-data](https://www.equinor.com/news/archive/14jun2018-disclosing-volve-data)

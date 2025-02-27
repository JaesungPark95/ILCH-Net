# ILCH-Net

Lithology Prediction Using ILCH-Net, LSTM, and AE Models
This repository contains code for predicting missing lithological data from well-log sequences using three different deep learning models: ILCH-Net, LSTM, and Autoencoder (AE). The dataset used in this study is the Volve dataset, which is available on Equinor's site.

Models
The following models are implemented in this repository:

1. ILCH-Net
ILCH-Net is a deep learning model designed to predict missing well-log data by combining Convolutional Variational Autoencoders (CVAE) with an iterative prediction approach. This model utilizes both geological information and well-log sequences to improve prediction accuracy.

Code: Model_ILCH-Net.py
2. LSTM
The LSTM model is used to predict missing data in time-series well-log sequences, leveraging Long Short-Term Memory (LSTM) networks to capture temporal dependencies.

Code: Model_LSTM.py
3. Autoencoder (AE)
The Autoencoder (AE) model is trained to learn a compressed representation of well-log sequences and predict missing data based on this representation.

Code: Model_AE.py
Data
The Volve dataset used in this study is available via the Equinor website:
https://www.equinor.com/news/archive/14jun2018-disclosing-volve-data

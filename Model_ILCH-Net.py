import collections
import copy
import os
import pickle
import random
from os.path import join
from pprint import pprint as pp

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Lambda, LeakyReLU, GaussianNoise, LSTM, Dropout, Flatten, Conv1D, Conv1DTranspose, Reshape, MaxPooling1D, UpSampling1D, Conv1DTranspose, Dropout
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
from keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, ks_2samp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed for reproducibility
SEED = 1004
np.random.seed(SEED)
random.seed(SEED)
rng = np.random.default_rng(SEED)
rfloat = rng.random()

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['KERAS_BACKEND'] = 'tensorflow'

tf.random.set_seed(SEED)

# Define mode and sequence size
mode = 6
# mode=0: sequence data making and normalization #mode=1: Onehot encoding #model=2: CVAE model training #mode=3: cvae prediction #mode=4: LSTM training #mode=5: prediction rock sequence using LSTM
# mode=6: Prediction of 2 missing data using ILCH-Net

seq = 20
Nan_Number = 0
test_well_num = 5 # well 0: 1A, well 1: 1B, well 2: 11A, well 3: 11 T2, well 4: 14, well 5: 4
well_feature_num = 5
rock_class = 3
latent_dim = 2 #for CVAE encoder output dimension(mu, var)
window_size=seq

load_path = '../data/volve/well_6/medfilter_class3_0201_outlier/'

# Set paths for saving models and figures
save_path_model = '../result/CVAE_LSTM_class3/save_litho_conv_modify_all_seq_mse_not_conv_iter_mean_new/seq'+str(seq)+'/'+str(test_well_num)+'/'
path_savefig = '../result/CVAE_LSTM_class3/figure_litho_conv_modify_all_seq_mse_not_conv_iter_mean_new/seq'+str(seq)+'/'+str(test_well_num)+'/'

def makedirs(path, path1):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path1)

makedirs(save_path_model, path_savefig)


def calculate_excluded_indices(W):
    if W % 2 == 1:
        excluded_front = excluded_back = (W - 1) // 2
    else:
        excluded_front = (W // 2) - 1
        excluded_back = W // 2

    return excluded_front, excluded_back

# Function to load sequence data for training, validation, and testing
def load_data():
    training=np.load(save_path_model+'Well num_'+str(test_well_num)+'_training_seq_'+str(seq)+'.npy')
    validation=np.load(save_path_model+'Well num_'+str(test_well_num)+'_validation_seq_'+str(seq)+'.npy')
    testing=np.load(save_path_model+'Well num_'+str(test_well_num)+'_testing_seq_'+str(seq)+'.npy')

    train_lithology_seq=np.load(save_path_model+'Well num_'+str(test_well_num)+'_training_lithology_seq_'+str(seq)+'.npy')
    validation_lithology_seq=np.load(save_path_model+'Well num_'+str(test_well_num)+'_validation_lithology_seq_'+str(seq)+'.npy')
    test_data_lithology_seq=np.load(save_path_model+'Well num_'+str(test_well_num)+'_testing_lithology_seq_'+str(seq)+'.npy')

    train_lithology_nan_seq=np.load(save_path_model+'Well num_'+str(test_well_num)+'_training_lithology_'+str(seq)+'.npy')
    validation_lithology_nan_seq=np.load(save_path_model+'Well num_'+str(test_well_num)+'_validation_lithology_'+str(seq)+'.npy')
    test_data_lithology_nan_seq=np.load(save_path_model+'Well num_'+str(test_well_num)+'_testing_lithology_'+str(seq)+'.npy')

    train_depth=np.load(save_path_model+'Well num_'+str(test_well_num)+'_training_depth_seq_'+str(seq)+'.npy')
    validation_depth=np.load(save_path_model + 'Well num_'+str(test_well_num)+'_validation_depth_seq_' + str(seq) + '.npy')
    test_depth=np.load(save_path_model + 'Well num_'+str(test_well_num)+'_testing_depth_seq_' + str(seq) + '.npy')

    return training,validation,testing,train_lithology_seq,validation_lithology_seq,test_data_lithology_seq,train_lithology_nan_seq,validation_lithology_nan_seq,test_data_lithology_nan_seq,train_depth,validation_depth,test_depth

# Sampling class for CVAE model
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Main function to apply Bayesian Optimization and train CVAE model
def CVAE_Bayesian(batch_size, n_epoch, dense1, dense2):
    tf.random.set_seed(42)
    batch_size = int(batch_size)
    n_epoch = int(n_epoch)
    dense1 = int(dense1)
    dense2 = int(dense2)

    ### Encoder ###
    encoder_inputs = Input(shape=(training_data_smote_res.shape[1]),name="encoder_inputs")
    condition_inputs = Input(shape=(training_cond.shape[1]),name="condition_inputs")
    concatenate_layer = concatenate([encoder_inputs, condition_inputs])
    dense1_layer = Dense(dense1, activation = 'relu',name="dense1")(concatenate_layer)
    encoder_dropout = Dropout(0.5)(dense1_layer)
    dense2_layer = Dense(dense2, activation='relu', name="dense2")(encoder_dropout)
    mu = Dense(latent_dim, activation='linear', name="z_mu")(dense2_layer)
    log_var = Dense(latent_dim, activation='linear', name="z_log_var")(dense2_layer)
    z = Sampling()([mu, log_var])
    z_cond = concatenate([z, condition_inputs])
    encoder = keras.Model(inputs=[encoder_inputs, condition_inputs], outputs=[mu, log_var, z_cond])

    ### Decoder ###
    latent_inputs = keras.Input(shape=(z_cond.shape[1],))
    decoder_dense1 = Dense(dense2,activation='relu',name="decoder_dense1")(latent_inputs)
    decoder_dropout = Dropout(0.5)(decoder_dense1)
    decoder_dense2 = Dense(dense1, activation='relu', name="decoder_dense2")(decoder_dropout)
    decoder_out = Dense(encoder_inputs.shape[1], activation='sigmoid', name="decoder_output")(decoder_dense2)
    decoder = keras.Model(latent_inputs, decoder_out, name='decoder')

    # Define CVAE model and its training loop
    class CVAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
            self.test_total_loss_tracker = keras.metrics.Mean(name='test_total_loss')
            self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
            self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.test_total_loss_tracker,
            ]

        def train_step(self, data):
            x_data, x_condition = data[0]
            with tf.GradientTape() as tape:
                train_mu, train_log_var, train_z_cond = self.encoder([x_data, x_condition])
                reconstruction = self.decoder(train_z_cond)
                reconstruction_loss = tf.reduce_mean(keras.losses.MSE(x_data, reconstruction))
                kl_loss = -0.5 * (1 + train_log_var - tf.square(train_mu) - tf.exp(train_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss * 1000 + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

        def test_step(self, data):
            test_data, test_condition = data[0]

            test_mu, test_log_var, test_z_cond = self.encoder([test_data, test_condition],training=False)
            test_reconstruction = self.decoder(test_z_cond, training=False)
            test_reconstruction_loss = tf.reduce_mean(keras.losses.MSE(test_data, test_reconstruction))
            kl_loss = -0.5 * (1 + test_log_var - tf.square(test_mu) - tf.exp(test_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            test_total_loss = test_reconstruction_loss * 1000 + kl_loss
            self.test_total_loss_tracker.update_state(test_total_loss)

            return {"test_loss": self.test_total_loss_tracker.result()}

    CVAE_model = CVAE(encoder, decoder)
    CVAE_model.compile(optimizer=Adam())

    CVAE_model.fit(x=[training_data_smote_res, training_cond], y= training_data_smote_res, shuffle=True, epochs=n_epoch, batch_size=batch_size, verbose=0)

    score=CVAE_model.evaluate(x=[validation_data, validation_cond], y= validation_data)

    return -1 * score

def CVAE_Bayesian_new(batch_size, n_epoch, dense1, dense2):
    tf.random.set_seed(42)
    batch_size = int(batch_size)
    n_epoch = int(n_epoch)
    dense1 = int(dense1)
    dense2 = int(dense2)

    ### Encoder ###
    encoder_inputs = Input(shape=(training_data_smote_res.shape[1]), name="encoder_inputs")
    condition_inputs = Input(shape=(training_cond.shape[1]), name="condition_inputs")
    concatenate_layer = concatenate([encoder_inputs, condition_inputs])
    dense1_layer = Dense(dense1, activation='relu', name="dense1")(concatenate_layer)
    encoder_dropout = Dropout(0.5)(dense1_layer)
    dense2_layer = Dense(dense2, activation='relu', name="dense2")(encoder_dropout)
    mu = Dense(latent_dim, activation='linear', name="z_mu")(dense2_layer)
    log_var = Dense(latent_dim, activation='linear', name="z_log_var")(dense2_layer)
    z = Sampling()([mu, log_var])
    z_cond = concatenate([z, condition_inputs])
    encoder = keras.Model(inputs=[encoder_inputs, condition_inputs], outputs=[mu, log_var, z_cond])

    ### Decoder ###
    latent_inputs = keras.Input(shape=(z_cond.shape[1],))
    decoder_dense1 = Dense(dense2, activation='relu', name="decoder_dense1")(latent_inputs)
    decoder_dropout = Dropout(0.5)(decoder_dense1)
    decoder_dense2 = Dense(dense1, activation='relu', name="decoder_dense2")(decoder_dropout)
    decoder_out = Dense(encoder_inputs.shape[1], activation='sigmoid', name="decoder_output")(decoder_dense2)
    decoder = keras.Model(latent_inputs, decoder_out, name='decoder')

    class CVAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
            self.test_total_loss_tracker = keras.metrics.Mean(name='test_total_loss')
            self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
            self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.test_total_loss_tracker,
            ]

        def train_step(self, data):
            x_data, x_condition = data[0]
            with tf.GradientTape() as tape:
                train_mu, train_log_var, train_z_cond = self.encoder([x_data, x_condition])
                reconstruction = self.decoder(train_z_cond)
                reconstruction_loss = tf.reduce_mean(keras.losses.MSE(x_data, reconstruction))
                kl_loss = -0.5 * (1 + train_log_var - tf.square(train_mu) - tf.exp(train_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss * 1000 + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

        def test_step(self, data):
            test_data, test_condition = data[0]
            test_mu, test_log_var, test_z_cond = self.encoder([test_data, test_condition], training=False)
            test_reconstruction = self.decoder(test_z_cond, training=False)
            test_reconstruction_loss = tf.reduce_mean(keras.losses.MSE(test_data, test_reconstruction))
            kl_loss = -0.5 * (1 + test_log_var - tf.square(test_mu) - tf.exp(test_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            test_total_loss = test_reconstruction_loss * 1000 + kl_loss
            self.test_total_loss_tracker.update_state(test_total_loss)

            return {"test_loss": self.test_total_loss_tracker.result()}

    CVAE_model=CVAE(encoder, decoder)
    CVAE_model.compile(optimizer=Adam())
    path_model = join(path_savefig + 'CVAE_structure_Lithology_seq' + str(seq) + '_encoder.png')
    tf.keras.utils.plot_model(CVAE_model.encoder, show_shapes=True, to_file=path_model, expand_nested=True)  # CVAE model Structure plot
    path_model = join(path_savefig + 'CVAE_structure_Lithology_seq' + str(seq) + '_decoder.png')
    tf.keras.utils.plot_model(CVAE_model.decoder, show_shapes=True, to_file=path_model, expand_nested=True)  # CVAE model Structure plot
    path_model = join(path_savefig + 'CVAE_structure_Lithology_seq' + str(seq) + '_cvae.png')
    tf.keras.utils.plot_model(CVAE_model, show_shapes=True, to_file=path_model, expand_nested=True)  # CVAE model Structure plot
    CVAE_model.fit(x=[training_data_smote_res, training_cond], y= training_data_smote_res, shuffle=True, epochs=n_epoch, batch_size=batch_size, verbose=0)
    score = CVAE_model.evaluate(x=[validation_data, validation_cond], y=validation_data)
    model_path = save_path_model + 'Well num_' + str(test_well_num) + '_CVAE_LSTM_bayesian_best_fit_seq' + str(
        seq) + '_encoder.h5'
    CVAE_model.encoder.save(model_path)
    model_path = save_path_model + 'Well num_' + str(test_well_num) + '_CVAE_LSTM_bayesian_best_fit_seq' + str(
        seq) + '_decoder.h5'
    CVAE_model.decoder.save(model_path)
    return -1 * score

if mode ==0:

    os.listdir(load_path)

    i=0
    for f in os.listdir(load_path):
        file_name=load_path+f
        if i == 0:
            data0=pd.read_csv(file_name)
            data0_lithology=data0.pop("Lithology")
            data0_lithology_val= data0.pop("Lithology_VAL")
        elif i==1:
            data1=pd.read_csv(file_name)
            data1_lithology = data1.pop("Lithology")
            data1_lithology_val = data1.pop("Lithology_VAL")
        elif i==2:
            data2=pd.read_csv(file_name)
            data2_lithology = data2.pop("Lithology")
            data2_lithology_val = data2.pop("Lithology_VAL")
        elif i == 3:
            data3 = pd.read_csv(file_name)
            data3_lithology = data3.pop("Lithology")
            data3_lithology_val = data3.pop("Lithology_VAL")
        elif i == 4:
            data4 = pd.read_csv(file_name)
            data4_lithology = data4.pop("Lithology")
            data4_lithology_val = data4.pop("Lithology_VAL")
        elif i == 5:
            data5 = pd.read_csv(file_name)
            data5_lithology = data5.pop("Lithology")
            data5_lithology_val = data5.pop("Lithology_VAL")
        i=i+1

    def sequence_data(input_data, Lithology_val):
        row=input_data.shape[0]
        output_data_row=row-2*seq
        column=input_data.shape[1]
        output_data_seq_row=1+seq*2

        data_new=np.zeros([output_data_row, output_data_seq_row, column])
        lithology_seq = np.zeros([output_data_row, seq])
        j = 0 #row
        for i in range(seq,row-seq):
            data_new[j,:,:]=input_data[i-seq:i+seq+1,:]
            j=j+1

        j = 0
        for i in range(seq, row - seq):
            k = 0
            for l in range(-seq, 0):
                lithology_seq[j, k] = Lithology_val[i + l]
                k = k + 1
            j = j + 1

        lithology_output=Lithology_val[seq:row-seq].to_numpy()

        output_data_new=data_new[~(np.isnan(data_new).any(axis=1).any(axis=1))]
        output_lithology_seq = lithology_seq[~(np.isnan(data_new).any(axis=1).any(axis=1))]
        output_lithology_output = lithology_output[~(np.isnan(data_new).any(axis=1).any(axis=1))]
        output_lithology_start = np.zeros(shape=(np.shape(output_lithology_output)))

        for i in range(0,seq):
            output_lithology_start[i]=seq-i
        return output_data_new, output_lithology_seq, output_lithology_output, output_lithology_start


    data0_dataset_seq, data0_lithology_seq, data0_lithology_output, data0_output_lithology_start = sequence_data(data0.to_numpy(),
                                                                                data0_lithology_val)
    data1_dataset_seq, data1_lithology_seq, data1_lithology_output, data1_output_lithology_start = sequence_data(data1.to_numpy(),
                                                                                data1_lithology_val)
    data2_dataset_seq, data2_lithology_seq, data2_lithology_output, data2_output_lithology_start = sequence_data(data2.to_numpy(),
                                                                                data2_lithology_val)
    data3_dataset_seq, data3_lithology_seq, data3_lithology_output, data3_output_lithology_start = sequence_data(data3.to_numpy(),
                                                                                data3_lithology_val)
    data4_dataset_seq, data4_lithology_seq, data4_lithology_output, data4_output_lithology_start = sequence_data(data4.to_numpy(),
                                                                                data4_lithology_val)
    data5_dataset_seq, data5_lithology_seq, data5_lithology_output, data5_output_lithology_start = sequence_data(data5.to_numpy(),
                                                                                data5_lithology_val)

    if test_well_num==0:
        train_data=np.concatenate((data1_dataset_seq,data2_dataset_seq,data3_dataset_seq,data4_dataset_seq,data5_dataset_seq),axis=0)
        test_data = data0_dataset_seq

        train_data_lithology_seq=np.concatenate((data1_lithology_seq,data2_lithology_seq,data3_lithology_seq,data4_lithology_seq,data5_lithology_seq),axis=0)
        test_data_lithology_seq=data0_lithology_seq

        train_data_lithology=np.concatenate((data1_lithology_output,data2_lithology_output,data3_lithology_output,data4_lithology_output,data5_lithology_output),axis=0)
        test_data_lithology=data0_lithology_output

        train_data_lithology_start=np.concatenate((data1_output_lithology_start,data2_output_lithology_start,data3_output_lithology_start,data4_output_lithology_start,data5_output_lithology_start),axis=0)
        test_data_lithology_start=data0_output_lithology_start

    elif test_well_num==1:
        train_data=np.concatenate((data0_dataset_seq,data2_dataset_seq,data3_dataset_seq,data4_dataset_seq,data5_dataset_seq),axis=0)
        test_data = data1_dataset_seq

        train_data_lithology_seq=np.concatenate((data0_lithology_seq,data2_lithology_seq,data3_lithology_seq,data4_lithology_seq,data5_lithology_seq),axis=0)
        test_data_lithology_seq=data1_lithology_seq

        train_data_lithology=np.concatenate((data0_lithology_output,data2_lithology_output,data3_lithology_output,data4_lithology_output,data5_lithology_output),axis=0)
        test_data_lithology=data1_lithology_output

        train_data_lithology_start=np.concatenate((data0_output_lithology_start,data2_output_lithology_start,data3_output_lithology_start,data4_output_lithology_start,data5_output_lithology_start),axis=0)
        test_data_lithology_start=data1_output_lithology_start

    elif test_well_num==2:
        train_data=np.concatenate((data0_dataset_seq,data1_dataset_seq,data3_dataset_seq,data4_dataset_seq,data5_dataset_seq),axis=0)
        test_data = data2_dataset_seq

        train_data_lithology_seq=np.concatenate((data0_lithology_seq,data1_lithology_seq,data3_lithology_seq,data4_lithology_seq,data5_lithology_seq),axis=0)
        test_data_lithology_seq=data2_lithology_seq

        train_data_lithology=np.concatenate((data0_lithology_output,data1_lithology_output,data3_lithology_output,data4_lithology_output,data5_lithology_output),axis=0)
        test_data_lithology=data2_lithology_output

        train_data_lithology_start=np.concatenate((data0_output_lithology_start,data1_output_lithology_start,data3_output_lithology_start,data4_output_lithology_start,data5_output_lithology_start),axis=0)
        test_data_lithology_start=data2_output_lithology_start

    elif test_well_num==3:
        train_data=np.concatenate((data0_dataset_seq,data1_dataset_seq,data2_dataset_seq,data4_dataset_seq,data5_dataset_seq),axis=0)
        test_data = data3_dataset_seq

        train_data_lithology_seq=np.concatenate((data0_lithology_seq,data1_lithology_seq,data2_lithology_seq,data4_lithology_seq,data5_lithology_seq),axis=0)
        test_data_lithology_seq=data3_lithology_seq

        train_data_lithology=np.concatenate((data0_lithology_output,data1_lithology_output,data2_lithology_output,data4_lithology_output,data5_lithology_output),axis=0)
        test_data_lithology=data3_lithology_output

        train_data_lithology_start=np.concatenate((data0_output_lithology_start,data1_output_lithology_start,data2_output_lithology_start,data4_output_lithology_start,data5_output_lithology_start),axis=0)
        test_data_lithology_start=data3_output_lithology_start

    elif test_well_num==4:
        train_data=np.concatenate((data0_dataset_seq,data1_dataset_seq,data2_dataset_seq,data3_dataset_seq,data5_dataset_seq),axis=0)
        test_data = data4_dataset_seq

        train_data_lithology_seq=np.concatenate((data0_lithology_seq,data1_lithology_seq,data2_lithology_seq,data3_lithology_seq,data5_lithology_seq),axis=0)
        test_data_lithology_seq=data4_lithology_seq

        train_data_lithology=np.concatenate((data0_lithology_output,data1_lithology_output,data2_lithology_output,data3_lithology_output,data5_lithology_output),axis=0)
        test_data_lithology=data4_lithology_output

        train_data_lithology_start=np.concatenate((data0_output_lithology_start,data1_output_lithology_start,data2_output_lithology_start,data3_output_lithology_start,data5_output_lithology_start),axis=0)
        test_data_lithology_start=data4_output_lithology_start

    elif test_well_num==5:
        train_data=np.concatenate((data0_dataset_seq,data1_dataset_seq,data2_dataset_seq,data3_dataset_seq,data4_dataset_seq),axis=0)
        test_data = data5_dataset_seq

        train_data_lithology_seq=np.concatenate((data0_lithology_seq,data1_lithology_seq,data2_lithology_seq,data3_lithology_seq,data4_lithology_seq),axis=0)
        test_data_lithology_seq=data5_lithology_seq

        train_data_lithology=np.concatenate((data0_lithology_output,data1_lithology_output,data2_lithology_output,data3_lithology_output,data4_lithology_output),axis=0)
        test_data_lithology=data5_lithology_output

        train_data_lithology_start=np.concatenate((data0_output_lithology_start,data1_output_lithology_start,data2_output_lithology_start,data3_output_lithology_start,data4_output_lithology_start),axis=0)
        test_data_lithology_start=data5_output_lithology_start

    train_data, validation_data, train_lithology, validation_lithology=train_test_split(train_data, np.concatenate((train_data_lithology_seq, train_data_lithology.reshape(-1,1), train_data_lithology_start.reshape(-1,1)), axis=1), train_size=0.7, shuffle=False) #using shuffle

    train_lithology_seq=train_lithology[:,:seq]
    train_lithology_nan_seq=train_lithology[:,seq:-1]
    train_lithology_start=train_lithology[:,-1]

    validation_lithology_seq=validation_lithology[:,:seq]
    validation_lithology_nan_seq=validation_lithology[:,seq:-1]
    validation_lithology_start = validation_lithology[:, -1]

    ### normalization train data ###
    train_data_normalization = train_data[:,seq,1:]
    MinMaxScaler_train_data = MinMaxScaler()
    train_data_log_MinMaxScaler = MinMaxScaler_train_data.fit_transform(train_data_normalization)

    ### normalization ###
    train_data_normalization_new = np.zeros((train_data.shape[0], train_data.shape[1], train_data.shape[2]))
    for i in range(0, train_data.shape[0]):
        train_data_normalization_new[i, :, 0] = train_data[i, :, 0]
        train_data_normalization_new[i, :, 1:] = MinMaxScaler_train_data.transform(train_data[i, :, 1:])

    validation_data_normalization_new = np.zeros((validation_data.shape[0], validation_data.shape[1], validation_data.shape[2]))
    for i in range(0, validation_data.shape[0]):
        validation_data_normalization_new[i, :, 0] = validation_data[i, :, 0]
        validation_data_normalization_new[i, :, 1:] = MinMaxScaler_train_data.transform(validation_data[i, :, 1:])

    test_data_normalization_new = np.zeros((test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    for i in range(0, test_data.shape[0]):
        test_data_normalization_new[i, :, 0] = test_data[i, :, 0]
        test_data_normalization_new[i, :, 1:] = MinMaxScaler_train_data.transform(test_data[i, :, 1:])

    MinMaxScaler_save_path = save_path_model + 'MinMaxScaler_save_seq_' + str(seq) + '.pkl'
    with open(MinMaxScaler_save_path, 'wb') as f:
        pickle.dump(MinMaxScaler_train_data, f)

    train_depth = train_data_normalization_new[:,seq,0]
    validation_depth = validation_data_normalization_new[:,seq,0]
    test_depth = test_data_normalization_new[:,seq,0]

    training = train_data_normalization_new[:,:,1:]
    validation = validation_data_normalization_new[:,:,1:]
    testing = test_data_normalization_new[:,:,1:]

    ### lithology_data_concatenate(previous and t depth) ###
    train_lithology_concatenate=np.concatenate((train_lithology_seq,train_lithology_nan_seq),axis=1)
    validation_lithology_concatenate=np.concatenate((validation_lithology_seq, validation_lithology_nan_seq),axis=1)
    test_lithology_concatenate=np.concatenate((test_data_lithology_seq, test_data_lithology.reshape(-1,1)),axis=1)

    plt.figure(figsize=(10,8))
    ax = sns.histplot(train_lithology_nan_seq)

    for p in ax.patches:
        if p.get_bbox().height > 0:
            left, bottom, width, height = p.get_bbox().bounds
            ax.annotate("%d" % (height), xy=(left + width / 2, bottom + height + 250), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(path_savefig + 'litho_percent_original_' + '.jpg', dpi=400, bbox_inches='tight', pad_inches=0.2)

    litho_num_1 = collections.Counter(train_lithology_nan_seq.reshape(-1))[1]
    litho_num_2 = collections.Counter(train_lithology_nan_seq.reshape(-1))[2]
    litho_num_3 = collections.Counter(train_lithology_nan_seq.reshape(-1))[3]

    #sequence data save
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_training_seq_'+str(seq)+'.npy',training)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_validation_seq_'+str(seq)+'.npy',validation)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_testing_seq_'+str(seq)+'.npy',testing)

    np.save(save_path_model+'Well num_'+str(test_well_num)+'_training_lithology_seq_'+str(seq)+'.npy',train_lithology_seq)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_validation_lithology_seq_'+str(seq)+'.npy',validation_lithology_seq)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_testing_lithology_seq_'+str(seq)+'.npy',test_data_lithology_seq)

    np.save(save_path_model+'Well num_'+str(test_well_num)+'_training_lithology_'+str(seq)+'.npy',train_lithology_nan_seq)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_validation_lithology_'+str(seq)+'.npy',validation_lithology_nan_seq)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_testing_lithology_'+str(seq)+'.npy',test_data_lithology)

    np.save(save_path_model+'Well num_'+str(test_well_num)+'_training_lithology_previous_concat_'+str(seq)+'.npy',train_lithology_concatenate)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_validation_lithology_previous_concat_'+str(seq)+'.npy',validation_lithology_concatenate)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_testing_lithology_previous_concat_'+str(seq)+'.npy',test_lithology_concatenate)

    np.save(save_path_model+'Well num_'+str(test_well_num)+'_training_lithology_start_seq_'+str(seq)+'.npy',train_lithology_start)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_validation_lithology_start_seq_'+str(seq)+'.npy',validation_lithology_start)
    np.save(save_path_model+'Well num_'+str(test_well_num)+'_testing_lithology_start_seq_'+str(seq)+'.npy',test_data_lithology_start)

    np.save(save_path_model+'Well num_'+str(test_well_num)+'_training_depth_seq_'+str(seq)+'.npy',train_depth)
    np.save(save_path_model + 'Well num_'+str(test_well_num)+'_validation_depth_seq_' + str(seq) + '.npy', validation_depth)
    np.save(save_path_model + 'Well num_'+str(test_well_num)+'_testing_depth_seq_' + str(seq) + '.npy', test_depth)

    training_res_1d = training.reshape(-1, training.shape[1] * training.shape[2])

  #### Oversampling_using_SMOTE_litho_training_data ####
    smote_sample = SMOTE(random_state=1004)
    training_data_smote, training_lithology_nan_seq_over = smote_sample.fit_resample(training_res_1d,train_lithology_nan_seq.reshape(-1, 1).astype(int))
    training_data_smote_res = training_data_smote.reshape(-1, training.shape[1],training.shape[2])
    plt.figure(figsize=(10,8))
    ax = sns.histplot(training_lithology_nan_seq_over)

    for p in ax.patches:
        if p.get_bbox().height > 0:
            left, bottom, width, height = p.get_bbox().bounds
            ax.annotate("%d" % (height), xy=(left + width / 2, bottom + height + 250), ha='center', va='center')

    plt.tight_layout()
    plt.savefig(path_savefig + 'litho_percent_CVAE_oversampling_' + '.jpg', dpi=400, bbox_inches='tight',pad_inches=0.2)
    litho_num_1_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[1]
    litho_num_2_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[2]
    litho_num_3_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[3]

    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy', training_data_smote_res)
    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_training_lithology_' + str(seq) + '_over.npy', training_lithology_nan_seq_over)

    smote_data_t = training_data_smote_res[:,seq,:]
    all_data_litho_con = np.concatenate((smote_data_t, training_lithology_nan_seq_over.reshape(-1, 1)), axis=1)
    all_data_pd = pd.DataFrame(all_data_litho_con, columns=['GR', 'NPHI', 'RHOB', 'DTC', 'DTS', 'Lithology'])
    all_data_pd['Lithology']=all_data_pd['Lithology'].replace([1],'claystone')
    all_data_pd['Lithology'] = all_data_pd['Lithology'].replace([2], 'sandstone')
    all_data_pd['Lithology'] = all_data_pd['Lithology'].replace([3], 'limestone')


elif mode == 1:
    np.seterr(divide='ignore', invalid='ignore')
    training, validation, testing, train_lithology_seq, validation_lithology_seq, test_lithology_seq, train_lithology_nan_seq, validation_lithology_nan_seq, test_lithology_nan_seq, train_depth, validation_depth, test_depth=load_data()
    training_lithology_nan_seq_over = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_training_lithology_' + str(seq) + '_over.npy')
    training_lithology_nan_seq_over = training_lithology_nan_seq_over.reshape(-1,1)

    enc = OneHotEncoder(sparse=False)
    enc.fit(training_lithology_nan_seq_over)
    joblib.dump(enc, save_path_model + 'OneHotEncoder.save')
    training_onehot=enc.transform(train_lithology_nan_seq)
    training_onehot_over=enc.transform(training_lithology_nan_seq_over)
    validation_onehot=enc.transform(validation_lithology_nan_seq)
    testing_onehot=enc.transform(test_lithology_nan_seq.reshape(-1,1))

    np.save(save_path_model + 'Well num_'+str(test_well_num)+'_training_onehot_seq_over_' + str(seq) + '.npy', training_onehot_over)
    np.save(save_path_model + 'Well num_'+str(test_well_num)+'_training_onehot_seq_' + str(seq) + '.npy', training_onehot)
    np.save(save_path_model + 'Well num_'+str(test_well_num)+'_validation_onehot_seq_' + str(seq) + '.npy', validation_onehot)
    np.save(save_path_model + 'Well num_'+str(test_well_num)+'_testing_onehot_seq_' + str(seq) + '.npy', testing_onehot)

elif mode == 2:
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()
    training_data_smote = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy')
    training_data_smote_res = training_data_smote.reshape(training_data_smote.shape[0], -1)
    validation_data = validation_data.reshape(validation_data.shape[0], -1)
    testing_data = testing_data.reshape(testing_data.shape[0], -1)

    training_cond = np.load(save_path_model + 'Well num_'+str(test_well_num)+'_training_onehot_seq_over_' + str(seq) + '.npy')
    validation_cond = np.load(save_path_model + 'Well num_'+str(test_well_num)+'_validation_onehot_seq_' + str(seq) + '.npy')

    max_dense1 = int(training_data_smote_res.shape[1])
    min_dense1 = int(max_dense1 * 0.5)
    max_dense2 = min_dense1
    min_dense2 = int(max_dense2 * 0.5)

    pbounds = {
        'batch_size': (1000, 2000),
        'n_epoch': (50, 500),
        'dense1': (min_dense1, max_dense1),
        'dense2': (min_dense2, max_dense2),
    }

    acc = []
    lossdata = []
    b0 = BayesianOptimization(f=CVAE_Bayesian, pbounds=pbounds, verbose=2, random_state=1004)
    b0.maximize(init_points=5, n_iter=5)
    print('\n==============================================================================')
    print("Bayesian Optimization parameter ")
    pp(b0.max)
    print('\n==============================================================================')

    with open( save_path_model+'Well num_'+str(test_well_num)+'_bayesian_result_CVAE_LSTM_seq' + str(seq) + '.pkl', 'wb') as f:
        pickle.dump(b0.max, f)

    fit_CVAE = CVAE_Bayesian_new(
        batch_size=b0.max['params']['batch_size'],
        n_epoch=b0.max['params']['n_epoch'],
        dense1=b0.max['params']['dense1'],
        dense2=b0.max['params']['dense2'],
    )

    print(fit_CVAE)

elif mode == 3:
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    training_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_onehot_seq_' + str(seq) + '.npy')
    validation_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_validation_onehot_seq_' + str(seq) + '.npy')
    testing_cond = np.load(
        save_path_model + 'Well num_'+str(test_well_num)+'_testing_onehot_seq_' + str(seq) + '.npy')
    training_cond_smote = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_onehot_seq_over_' + str(seq) + '.npy')
    training_data_smote = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy')
    cvae_encoder=keras.models.load_model(save_path_model + 'Well num_' + str(test_well_num) + '_CVAE_LSTM_bayesian_best_fit_seq' + str(
        seq) + '_encoder.h5', custom_objects={"Sampling": Sampling,})

    training_data_smote_res = training_data_smote.reshape(training_data_smote.shape[0], -1)
    training_data = training_data.reshape(training_data.shape[0],-1)
    validation_data = validation_data.reshape(validation_data.shape[0], -1)
    testing_data = testing_data.reshape(testing_data.shape[0], -1)

    cvae_decoder = keras.models.load_model(
        save_path_model + 'Well num_' + str(test_well_num) + '_CVAE_LSTM_bayesian_best_fit_seq' + str(
            seq) + '_decoder.h5')

    load_MinMaxScaler_scaler = pickle.load(open(save_path_model + 'MinMaxScaler_save_seq_' + str(seq) + '.pkl', 'rb'))

    training_mu, training_log_var, training_z_cond = cvae_encoder.predict([training_data, training_cond])
    pred_training = cvae_decoder.predict(training_z_cond)

    validation_mu, validation_log_var, validation_z_cond = cvae_encoder.predict([validation_data, validation_cond])
    pred_validation = cvae_decoder.predict(validation_z_cond)

    testing_mu, testing_log_var, testing_z_cond = cvae_encoder.predict([testing_data, testing_cond])
    pred_testing = cvae_decoder.predict(testing_z_cond)

    training_mu_smote, training_log_var_smote, training_z_cond_smote = cvae_encoder.predict([training_data_smote_res, training_cond_smote])
    pred_training_smote = cvae_decoder.predict(training_z_cond_smote)

    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_training_seq_' + str(seq) + '.npy',
            pred_training)
    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_validation_seq_' + str(seq) + '.npy',
            pred_validation)
    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_testing_seq_' + str(seq) + '.npy',
            pred_testing)
    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_training_seq_over_' + str(seq) + '.npy',
            pred_training_smote)

elif mode == 4: #LSTM training
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    training_data_smote_res = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy')

    pred_training_smote = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_pred_training_seq_over_' + str(seq) + '.npy')

    pred_training_smote = pred_training_smote.reshape(-1, seq * 2 + 1, well_feature_num)

    training_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_onehot_seq_over_' + str(seq) + '.npy')
    validation_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_validation_onehot_seq_' + str(seq) + '.npy')
    testing_cond = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_testing_onehot_seq_' + str(seq) + '.npy')

    pred_training_smote_prev = pred_training_smote #[:,:seq+1,:]

    pred_training_prev = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_pred_training_seq_' + str(seq) + '.npy')
    pred_training_prev = pred_training_prev.reshape(-1, seq * 2 + 1, well_feature_num)

    pred_validation_prev = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_pred_validation_seq_' + str(seq) + '.npy')
    pred_validation_prev = pred_validation_prev.reshape(-1, seq * 2 + 1, well_feature_num)

    pred_testing_prev = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_pred_testing_seq_' + str(seq) + '.npy')
    pred_testing_prev = pred_testing_prev.reshape(-1, seq * 2 + 1, well_feature_num)

    def LSTM_test(batch_size, n_epoch, n_hidden1, n_hidden2, dropout_rate):
        tf.random.set_seed(42)

        batch_size = int(batch_size)
        n_epoch = int(n_epoch)
        n_hidden1 = int(n_hidden1)
        n_hidden2 = int(n_hidden2)
        dropout_rate = float(dropout_rate)

        model = Sequential()
        model.add(LSTM(n_hidden1, input_shape=(seq*2+1, well_feature_num), return_sequences=False))
        model.add(Flatten())
        model.add(Dense(n_hidden2, activation='tanh'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(rock_class, activation='softmax'))
        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['CategoricalAccuracy'])
        model.fit(pred_training_smote_prev, training_cond, batch_size=batch_size, epochs=n_epoch, validation_data=(pred_validation_prev, validation_cond),verbose=0)

        score=model.evaluate(pred_validation_prev, validation_cond)

        return score[0] * -1

    pbounds={
        'batch_size': (1000, 2000),
        'n_epoch': (1, 200),
        'n_hidden1': (10, 40),
        'n_hidden2': (9, 30),
        'dropout_rate': (0.1, 0.5),
    }

    b1=BayesianOptimization(f=LSTM_test, pbounds=pbounds, verbose=2, random_state=1004)
    b1.maximize(init_points=10, n_iter=10)
    print('\n==============================================================================')
    print("Bayesian Optimization parameter")
    pp(b1.max)
    print('\n==============================================================================')


    def LSTM_test_Optimize(batch_size, n_epoch, n_hidden1, n_hidden2, dropout_rate):
        tf.random.set_seed(42)

        batch_size = int(batch_size)
        n_epoch = int(n_epoch)
        n_hidden1 = int(n_hidden1)
        n_hidden2 = int(n_hidden2)
        dropout_rate = float(dropout_rate)

        model = Sequential()
        model.add(LSTM(n_hidden1, input_shape=(seq*2+1, well_feature_num), return_sequences=False))
        model.add(Flatten())
        model.add(Dense(n_hidden2, activation='tanh'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(rock_class, activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['CategoricalAccuracy'])
        model.summary()

        path_model = join(path_savefig + 'LSTM_structure_Lithology_seq' + str(seq) + '.png')
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=path_model)

        history_curve=model.fit(pred_training_smote_prev, training_cond, batch_size=batch_size, epochs=n_epoch,
                  validation_data=(pred_validation_prev, validation_cond),verbose=0)

        fig, loss_ax = plt.subplots()
        loss_ax.plot(history_curve.history['loss'], 'b', label='train loss')
        loss_ax.plot(history_curve.history['val_loss'], 'g', label='validation loss')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper right')

        plt.savefig(path_savefig + 'LSTM_learning_curve.jpg', dpi=400, bbox_inches='tight', pad_inches=0.2)

        score = model.evaluate(pred_validation_prev, validation_cond)

        model.save(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_seq_' + str(seq) + '.h5')

        train_score = model.evaluate(pred_training_smote_prev, training_cond)
        print(train_score)

        validation_score = model.evaluate(pred_validation_prev, validation_cond)
        print(validation_score)

        testing_score = model.evaluate(pred_testing_prev, testing_cond)
        print(testing_score)

        test_predictions = model.predict(pred_testing_prev)
        np.save(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_test_result_seq_' + str(seq) + '.npy',
                test_predictions)

        training_predictions = model.predict(pred_training_smote_prev)
        np.save(
            save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_training_result_seq_' + str(seq) + '.npy',
            training_predictions)

        validation_predictions = model.predict(pred_validation_prev)
        np.save(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_validation_result_seq_' + str(
            seq) + '.npy',
                validation_predictions)

        return score[0] * -1

    fit_LSTM=LSTM_test_Optimize(
        batch_size=b1.max['params']['batch_size'],
        n_epoch=b1.max['params']['n_epoch'],
        n_hidden1=b1.max['params']['n_hidden1'],
        n_hidden2=b1.max['params']['n_hidden2'],
        dropout_rate=b1.max['params']['dropout_rate'])

    print(fit_LSTM)

elif mode == 5:
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    pred_training_prev = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_pred_training_seq_' + str(seq) + '.npy')
    pred_training_prev = pred_training_prev.reshape(-1, seq * 2 + 1, well_feature_num)
    pred_validation_prev = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_pred_validation_seq_' + str(seq) + '.npy')
    pred_validation_prev = pred_validation_prev.reshape(-1, seq * 2 + 1, well_feature_num)
    pred_testing_prev = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_pred_testing_seq_' + str(seq) + '.npy')
    pred_testing_prev = pred_testing_prev.reshape(-1, seq * 2 + 1, well_feature_num)
    training_data_smote_res = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy')
    training_data_smote_res = training_data_smote_res.reshape(-1, seq * 2 + 1, well_feature_num)

    training_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_onehot_seq_over_' + str(seq) + '.npy')
    validation_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_validation_onehot_seq_' + str(seq) + '.npy')
    testing_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_testing_onehot_seq_' + str(seq) + '.npy')

    model = tf.keras.models.load_model(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_seq_' + str(seq) + '.h5')

    training_predictions=model.predict(training_data_smote_res)
    validation_predictions=model.predict(pred_validation_prev)
    test_predictions=model.predict(pred_testing_prev)

    train_score = model.evaluate(training_data_smote_res, training_cond)
    print(train_score)

    validation_score = model.evaluate(pred_validation_prev, validation_cond)
    print(validation_score)

    testing_score = model.evaluate(pred_testing_prev, testing_cond)
    print(testing_score)

elif mode == 6:
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    training_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_onehot_seq_' + str(seq) + '.npy')
    validation_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_validation_onehot_seq_' + str(seq) + '.npy')
    testing_cond = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_testing_onehot_seq_' + str(seq) + '.npy')

    ####LSTM Load####
    load_MinMaxScaler_scaler = pickle.load(open(save_path_model + 'MinMaxScaler_save_seq_' + str(seq) + '.pkl', 'rb'))
    model = tf.keras.models.load_model(
        save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_seq_' + str(seq) + '.h5')

    title = ['GR', 'NPHI', 'RHOB', 'DTC', 'DTS']
    column_name = ["Perason CC","Sperman CC", "Spearman p-val", "RMSE", "R2","KS D" ,"KS p-val"] * (well_feature_num-1)
    matrix = np.zeros((well_feature_num, 7 * (well_feature_num - 1)))

    row_name = [[0 for l in range(well_feature_num-1)] for j in range(well_feature_num)]


    k = 0
    iteration = 100
    patience = 5

    pred_test_NaN = np.zeros((testing_data.shape[0], testing_data.shape[1], testing_data.shape[2]))
    pred_test_subtract_previous = np.zeros((iteration,1))

    patience_count = 0

    front_idx, back_idx = calculate_excluded_indices(window_size)

    for i in range(0, well_feature_num-1):
        # test nan data plot
        x_test_data_NaN = copy.deepcopy(testing_data)
        x_test_data_NaN[:, :, i] = np.random.rand(x_test_data_NaN.shape[0],x_test_data_NaN.shape[1])
        x_test_data_NaN[:, :, 4] = np.random.rand(x_test_data_NaN.shape[0],x_test_data_NaN.shape[1]) #DTS

        test_initial_cond=np.full((testing_cond.shape[0],testing_cond.shape[1]),1/3)

        ####CVAE Load####
        cvae_encoder = keras.models.load_model(
            save_path_model + 'Well num_' + str(test_well_num) + '_CVAE_LSTM_bayesian_best_fit_seq' + str(
                seq) + '_encoder.h5', custom_objects={"Sampling": Sampling, })
        cvae_decoder = keras.models.load_model(
            save_path_model + 'Well num_' + str(test_well_num) + '_CVAE_LSTM_bayesian_best_fit_seq' + str(
                seq) + '_decoder.h5')
        x_test_data_NaN = x_test_data_NaN.reshape(x_test_data_NaN.shape[0],-1)
        testing_mu, testing_log_var, testing_z_cond = cvae_encoder.predict([x_test_data_NaN, test_initial_cond])
        pred_test_NaN = cvae_decoder.predict(testing_z_cond)
        pred_test_NaN = pred_test_NaN.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2])
        x_test_data_NaN = x_test_data_NaN.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2])
        x_test_data_NaN[:,:,i] = pred_test_NaN[:,:,i]
        x_test_data_NaN[:, :, 4] = pred_test_NaN[:, :, 4]

        x_test_data_NaN_seq = x_test_data_NaN
        testing_seq_data_pred = model.predict(x_test_data_NaN_seq)
        pred_test_NaN_seq = pred_test_NaN[:, seq, :]
        testing_data_seq = testing_data[:, seq, :]
        testing_score_curve=np.zeros((iteration,2))
        testing_nan_data=np.zeros((iteration,2))
        testing_nan_dts_data=np.zeros((iteration,2))

        moving_averages_pred_nan = np.zeros(
            (pred_test_NaN_seq.shape[0] - window_size + 1, pred_test_NaN_seq.shape[1]))
        org_x_test_data_seq = testing_data_seq[front_idx:-back_idx, :]

        for s in range(pred_test_NaN_seq.shape[1]):
            moving_averages_pred_nan[:, s] = np.convolve(pred_test_NaN_seq[:, s],
                                                         np.ones(window_size) / window_size, mode='valid')

        testing_nan_dts_data[0, 0] = np.round(np.corrcoef(moving_averages_pred_nan[:, 4], org_x_test_data_seq[:, 4])[0, 1],
                                              2)
        testing_nan_dts_data[0, 1] = np.round(
            np.sqrt(mean_squared_error(load_MinMaxScaler_scaler.inverse_transform(moving_averages_pred_nan)[:, 4],
                                       load_MinMaxScaler_scaler.inverse_transform(org_x_test_data_seq)[:, 4])), 2)  ##RMSE

        testing_nan_data[0, 0] = np.round(np.corrcoef(moving_averages_pred_nan[:, i], org_x_test_data_seq[:, i])[0, 1], 2)  ##CC
        testing_nan_data[0, 1] = np.round(np.sqrt(mean_squared_error(moving_averages_pred_nan[:, i],
                                                                     load_MinMaxScaler_scaler.inverse_transform(
                                                                         org_x_test_data_seq)[:, i])), 2)  ##RMSE

        testing_score_curve[0, :] = model.evaluate(x_test_data_NaN_seq, testing_cond)

        prev_pred_test_NaN_seq = pred_test_NaN_seq

        # Iteration prediction
        for m in range(1, iteration):
            x_test_data_NaN = x_test_data_NaN.reshape(x_test_data_NaN.shape[0], -1)
            testing_mu, testing_log_var, testing_z_cond = cvae_encoder.predict([x_test_data_NaN,testing_seq_data_pred])
            pred_test_NaN = cvae_decoder.predict(testing_z_cond)

            pred_test_NaN = pred_test_NaN.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2])
            x_test_data_NaN = x_test_data_NaN.reshape(testing_data.shape[0], testing_data.shape[1],
                                                      testing_data.shape[2])
            x_test_data_NaN[:, :, i] = pred_test_NaN[:, :, i]
            x_test_data_NaN[:, :, 4] = pred_test_NaN[:, :, 4]

            pred_test_NaN_seq=pred_test_NaN[:,seq,:]
            testing_data_seq=testing_data[:,seq,:]

            moving_averages_pred_nan = np.zeros(
                (pred_test_NaN_seq.shape[0] - window_size + 1, pred_test_NaN_seq.shape[1]))

            for s in range(pred_test_NaN_seq.shape[1]):
                moving_averages_pred_nan[:, s] = np.convolve(pred_test_NaN_seq[:, s],
                                                             np.ones(window_size) / window_size, mode='valid')
            org_x_test_data_seq = testing_data_seq[front_idx:-back_idx, :]
            testing_nan_data[m, 0] = np.round(np.corrcoef(moving_averages_pred_nan[:,i], org_x_test_data_seq[:,i])[0,1],2) ##CC
            testing_nan_data[m, 1] = np.round(np.sqrt(mean_squared_error(moving_averages_pred_nan[:, i], load_MinMaxScaler_scaler.inverse_transform(org_x_test_data_seq)[:, i])), 2) ##RMSE

            testing_nan_dts_data[m, 0] = np.round(np.corrcoef(moving_averages_pred_nan[:, 4], org_x_test_data_seq[:, 4])[0, 1],
                                              2) ##CC
            testing_nan_dts_data[m, 1] = np.round(
                np.sqrt(mean_squared_error(load_MinMaxScaler_scaler.inverse_transform(moving_averages_pred_nan)[:, 4], load_MinMaxScaler_scaler.inverse_transform(org_x_test_data_seq)[:, 4])), 2)  ##RMSE

            x_test_data_NaN_seq = x_test_data_NaN

            testing_seq_data_pred = model.predict(x_test_data_NaN_seq)

            testing_score_curve[m, :] = model.evaluate(x_test_data_NaN_seq, testing_cond)

            pred_test_subtract_previous[m] = sum(sum(abs(pred_test_NaN_seq - prev_pred_test_NaN_seq)))

            prev_pred_test_NaN_seq = pred_test_NaN_seq

            if m > 1:
                if pred_test_subtract_previous[m - 1] < pred_test_subtract_previous[m]:
                    patience_count = patience_count + 1
                    print(
                        f"Difference value Previous<Current : {pred_test_subtract_previous[m]}. Iteration: {m}. Patience count: {patience_count} ")
                    if patience_count == patience:
                        print(
                            f"Patience : {pred_test_subtract_previous[m]}. Iteration: {m}. Patience count: {patience_count} ")
                        break
                else:
                    print(f"current value: {np.round(pred_test_subtract_previous[m], 2)}. Iteration: {m}")
            elif m == 1:
                patience_count = 0
                print(f"current value: {np.round(pred_test_subtract_previous[m], 2)}. Iteration: {m}")

        ## testing_nan_curve_plot CC
        plt.figure(figsize=(11, 10))
        tim = range(1, m + 1)
        plt.plot(tim, testing_nan_data[:m, 0])
        plt.xlabel('iteration', fontsize=10)
        plt.ylabel('CC', fontsize=10)
        plt.title('Prediction_NAN_'+str(title[k])+'_Iteration: ' + str(m), fontsize=15, fontweight='bold')
        path_savefig_testing_litho = path_savefig + 'testing_litho_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_NaN_CC_iteration_' + str(m) + '_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_litho, dpi=400, bbox_inches='tight', pad_inches=0.2)

        ## testing_nan_curve_plot RMSE
        plt.figure(figsize=(11, 10))
        tim = range(1, m + 1)
        plt.plot(tim, testing_nan_data[:m, 1])
        plt.xlabel('iteration', fontsize=10)
        plt.ylabel('RMSE', fontsize=10)
        plt.title('Prediction_NAN_' + str(title[k]) + '_Iteration: ' + str(m), fontsize=15, fontweight='bold')
        path_savefig_testing_litho = path_savefig + 'testing_litho_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_NaN_RMSE_iteration_' + str(m) + '_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_litho, dpi=400, bbox_inches='tight', pad_inches=0.2)

        ## testing_dts_curve_plot_CC
        plt.figure(figsize=(11, 10))
        tim = range(1, m + 1)
        plt.plot(tim, testing_nan_dts_data[:m, 0])
        plt.xlabel('iteration', fontsize=10)
        plt.ylabel('CC', fontsize=10)
        plt.title('Prediction_NAN_DTS_'+str(title[k])+'_Iteration: ' + str(m), fontsize=15, fontweight='bold')
        path_savefig_testing_litho = path_savefig + 'testing_litho_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_NaN_dts_CC_iteration_' + str(m) + '_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_litho, dpi=400, bbox_inches='tight', pad_inches=0.2)

        ## testing_dts_curve_plot_RMSE
        plt.figure(figsize=(11, 10))
        tim = range(1, m + 1)
        plt.plot(tim, testing_nan_dts_data[:m, 1])
        plt.xlabel('iteration', fontsize=10)
        plt.ylabel('RMSE', fontsize=10)
        plt.title('Prediction_NAN_DTS_'+str(title[k])+'_Iteration: ' + str(m), fontsize=15, fontweight='bold')
        path_savefig_testing_litho = path_savefig + 'testing_litho_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_NaN_dts_RMSE_iteration_' + str(m) + '_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_litho, dpi=400, bbox_inches='tight', pad_inches=0.2)

        ## testing_litho_curve_plot
        plt.figure(figsize=(11, 10))
        tim=range(1,m+1)
        plt.plot(tim,np.round(testing_score_curve[:m,1]*100,1))
        plt.xlabel('iteration', fontsize=10)
        plt.ylabel('acc', fontsize=10)
        plt.title('Litho_Iteration: '+str(m), fontsize=15, fontweight='bold')
        path_savefig_testing_litho = path_savefig + 'testing_litho_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_iteration_'+str(m)+'_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_litho, dpi=400, bbox_inches='tight', pad_inches=0.2)

        pred_test_subtract_previous=np.abs(pred_test_subtract_previous)

        pred_test_subtract_previous = pred_test_subtract_previous[1:m + 1]

        pred_test_subtract_previous_pd = pd.DataFrame(pred_test_subtract_previous)

        result_path = save_path_model + 'testing_welllog_NaN_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred_iteration_subtract.xlsx'

        with pd.ExcelWriter(result_path) as writer:
            pred_test_subtract_previous_pd.to_excel(writer, sheet_name='result')

        ## testing_subtract_iteration
        plt.figure(figsize=(11, 10))
        tim = range(1, m + 1)
        plt.plot(tim, np.round(pred_test_subtract_previous[:m], 4), color='black')
        plt.xlabel('Iteration', fontweight='bold', fontsize=15)
        plt.ylabel('Value', fontweight='bold', fontsize=15)
        plt.xticks(ticks=tim, fontweight='bold', fontsize=12)
        plt.yticks(fontweight='bold',fontsize=12)
        path_savefig_testing_litho = path_savefig + 'testing_litho_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_subtraction_iteration_' + str(m) + '_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_litho, dpi=400, bbox_inches='tight', pad_inches=0.2)

        print("=================================================================")
        testing_score = model.evaluate(x_test_data_NaN_seq, testing_cond)

        print(testing_score)
        print("=================================================================")

        pred_test_NaN = pred_test_NaN.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2])
        x_test_data_NaN = x_test_data_NaN.reshape(testing_data.shape[0], testing_data.shape[1],
                                                  testing_data.shape[2])

        # testing_litho plot
        plt.figure(figsize=(11, 10))
        #plt.gca()
        fig, axs = plt.subplots(1,5, figsize=(11, 10),constrained_layout=True)
        plt_testing = np.repeat(np.argmax(testing_cond, axis=1), 500).reshape(testing_cond.shape[0], 500)

        axs[0].imshow(plt_testing, vmin=0, vmax=rock_class - 1)

        testing_labels = []
        testing_labels_locs = []
        depth_testing=testing_depth.reshape(-1)
        for i in range(0, depth_testing.shape[0]):
            if (depth_testing[i] % 100) == 0:
                testing_labels.append(str(depth_testing[i]))
                testing_labels_locs.append(i)

        axs[0].set_yticks(testing_labels_locs)
        axs[0].set_yticklabels(testing_labels)
        axs[0].set_xticks([0, 500])
        axs[0].set_xticklabels(["", ""])
        axs[0].set_title('True', fontsize=10)

        plt_pred_testing = np.repeat(np.argmax(testing_seq_data_pred, axis=1), 500).reshape(
            testing_seq_data_pred.shape[0], 500)

        axs[1].imshow(plt_pred_testing, vmin=0, vmax=rock_class - 1)
        axs[1].set_yticks([0, 10])
        axs[1].set_yticklabels(["", ""])
        axs[1].set_xticks([0, 500])
        axs[1].set_xticklabels(["", ""])
        axs[1].set_title('Prediction(arg_max)', fontsize=10)

        plt_pred_testing_class0 = np.repeat(testing_seq_data_pred[:, 0], 500).reshape(testing_seq_data_pred.shape[0], 500)
        axs[2].imshow(plt_pred_testing_class0, vmin=0, vmax=1, cmap='YlGnBu')
        axs[2].set_yticks([0, 10])
        axs[2].set_yticklabels(["", ""])
        axs[2].set_xticks([0, 500])
        axs[2].set_xticklabels(["", ""])
        axs[2].set_title('Probability(class 0)', fontsize=10, color='b')

        plt_pred_testing_class1 = np.repeat(testing_seq_data_pred[:, 1], 500).reshape(testing_seq_data_pred.shape[0],
                                                                                      500)
        axs[3].imshow(plt_pred_testing_class1, vmin=0, vmax=1, cmap='YlGnBu')
        axs[3].set_yticks([0, 10])
        axs[3].set_yticklabels(["", ""])
        axs[3].set_xticks([0, 500])
        axs[3].set_xticklabels(["", ""])
        axs[3].set_title('Probability(class 1)', fontsize=10, color='b')

        plt_pred_testing_class2 = np.repeat(testing_seq_data_pred[:, 2], 500).reshape(testing_seq_data_pred.shape[0], 500)
        cl2=axs[4].imshow(plt_pred_testing_class2, vmin=0, vmax=1, cmap='YlGnBu')
        axs[4].set_yticks([0, 10])
        axs[4].set_yticklabels(["", ""])
        axs[4].set_xticks([0, 500])
        axs[4].set_xticklabels(["", ""])
        axs[4].set_title('Probability(class 2)', fontsize=10, color='b')

        fig.colorbar(cl2, ax=axs)
        plt.suptitle('Testing data : '+str(np.round(testing_score[1]*100,2))+"% ", fontsize=15)
        path_savefig_testing_litho = path_savefig + 'testing_litho_NaN_'+title[k]+'_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_litho, dpi=400, bbox_inches='tight', pad_inches=0.2)

        pred_x_test_data_new = pred_test_NaN.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2])
        x_test_data_NaN = x_test_data_NaN.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2])

        ####### missing data stats save ######
        #testing_concatenate
        testing_mu_litho = np.concatenate((testing_mu, testing_Lithology_val_new.reshape(-1, 1)), axis=1)
        testing_sig_litho = np.concatenate((testing_log_var, testing_Lithology_val_new.reshape(-1, 1)), axis=1)

        testing_mu_litho_pd = pd.DataFrame(testing_mu_litho)
        testing_sig_litho_pd = pd.DataFrame(testing_sig_litho)

        # testing_mu_total_stats
        testing_mu_litho_total_stats_path = (
                save_path_model + 'Well num_' + str(test_well_num) + '_testing_well_log_NaN'+title[k]+'_mu_total_stats_' + str(seq) + '.csv')
        (testing_mu_litho_pd.describe()).to_csv(testing_mu_litho_total_stats_path)

        # testing_mu_litho_stats
        testing_mu_litho_litho_stats_path = (
                save_path_model + 'Well num_' + str(test_well_num) + '_testing_well_log_NaN'+title[k]+'_mu_litho_stats_' + str(seq) + '.csv')
        (testing_mu_litho_pd.groupby(2).describe()).to_csv(testing_mu_litho_litho_stats_path)

        # testing_sig_total_stats
        testing_sig_litho_total_stats_path = (
                save_path_model + 'Well num_' + str(test_well_num) + '_testing_well_log_NaN'+title[k]+'_sig_total_stats_' + str(seq) + '.csv')
        (testing_sig_litho_pd.describe()).to_csv(testing_sig_litho_total_stats_path)

        # testing_sig_litho_stats
        testing_sig_litho_litho_stats_path = (
                save_path_model + 'Well num_' + str(test_well_num) + '_testing_well_log_NaN'+title[k]+'_sig_litho_stats_' + str(seq) + '.csv')
        (testing_sig_litho_pd.groupby(2).describe()).to_csv(testing_sig_litho_litho_stats_path)

        hue_order = ['2', '3', '1']  # sandstone, limestone claystone

        # testing mu histogram litho
        fig = plt.figure(figsize=(50, 20))
        fig.suptitle("Testing_NAN_"+str(title[k])+"_mu_hist", fontsize=40, fontweight='bold', position=(0.5, 0.95))

        # First subplot
        ax1 = plt.subplot(1, 2, 1)
        mu_hist0 = sns.histplot(testing_mu_litho_pd, x=0, hue=2, hue_order=hue_order, bins=50, kde=True,
                                palette='Spectral', ax=ax1)
        mu_hist0.set_xlabel('X Label', fontsize=30)  # Replace 'X Label' with your x-axis label
        mu_hist0.set_ylabel('Y Label', fontsize=30)  # Replace 'Y Label' with your y-axis label
        mu_hist0.set_title('First Subplot Title', fontsize=35)
        legend = mu_hist0.get_legend()
        plt.setp(legend.get_texts(), fontsize=30)  # Set legend text size
        plt.setp(legend.get_title(), fontsize=30)  # Set legend title size

        # Second subplot
        ax2 = plt.subplot(1, 2, 2)
        mu_hist1 = sns.histplot(testing_mu_litho_pd, x=1, hue=2, hue_order=hue_order, bins=50, kde=True,
                                palette='Spectral', ax=ax2)
        mu_hist1.set_xlabel('X Label', fontsize=30)  # Replace 'X Label' with your x-axis label
        mu_hist1.set_ylabel('Y Label', fontsize=30)  # Replace 'Y Label' with your y-axis label
        mu_hist1.set_title('Second Subplot Title', fontsize=35)
        legend = mu_hist1.get_legend()
        plt.setp(legend.get_texts(), fontsize=30)  # Set legend text size
        plt.setp(legend.get_title(), fontsize=30)  # Set legend title size

        # Update the tick parameters for all axes
        ax1.tick_params(labelsize=25)
        ax2.tick_params(labelsize=25)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path_savefig + 'prediction_CVAE_testing_NaN_' + str(title[k]) + '_mu_hist_litho'+'.jpg', dpi=400, bbox_inches='tight',
                    pad_inches=0.2)

        org_x_test_data_new = testing_data
        org_x_test_data_new = org_x_test_data_new[:, seq,:]
        org_x_test_data_new = load_MinMaxScaler_scaler.inverse_transform(org_x_test_data_new)

        pred_x_test_data_result = pred_x_test_data_new
        pred_x_test_data_new = pred_x_test_data_new[:, seq,:]

        ## Mean trend
        moving_averages_pred_nan = np.zeros((pred_x_test_data_new.shape[0] - window_size + 1, pred_x_test_data_new.shape[1]))

        for s in range(pred_x_test_data_new.shape[1]):  # 각 열에 대하여
            moving_averages_pred_nan[:, s] = np.convolve(pred_x_test_data_new[:, s], np.ones(window_size) / window_size, mode='valid')

        pred_x_test_data_new_mean=moving_averages_pred_nan
        org_x_test_data_new_mean=org_x_test_data_new[front_idx:-back_idx,:]

        ####### missing data figure plot ######
        testing_labels = []
        testing_labels_locs = []
        depth_testing=testing_depth[front_idx:-back_idx].reshape(-1)
        for i in range(0, depth_testing.shape[0]):
            if (depth_testing[i] % 100) == 0:
                testing_labels.append(str(depth_testing[i]))
                testing_labels_locs.append(i)

        pred_x_test_data_new_mean = load_MinMaxScaler_scaler.inverse_transform(pred_x_test_data_new_mean)

        fig = plt.figure(figsize=(25, 30))
        tim = np.linspace(1, org_x_test_data_new_mean.shape[0], org_x_test_data_new_mean.shape[0])
        title_name = 'Test_NAN ' + title[k]
        fig.suptitle(title_name, fontsize=40, fontweight='bold', position=(0.5, 1))

        for j in range(0, 5):
            plt.subplot(1, 5, j + 1)
            if k == j:
                ax = plt.gca()
                plt.plot(org_x_test_data_new_mean[:, j], tim[:], color='k', label='Actual', linewidth=2)
                plt.plot(pred_x_test_data_new_mean[:, j], tim[:], color='r', label='Pred', linewidth=2)

                print('NAN ' + title[k] + ' Pearson CC:' + str(
                    np.round(np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1], 2))) # perason
                print('NAN ' + title[k] + ' Spearman CC:' + str(
                    np.round(spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0],
                             2)))  # spearman
                print('NAN ' + title[k] + ' Spearman p-val:' + str(
                    np.round(spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1],
                             2)))  # spearman p-value
                print('NAN ' + title[k] + ' RMSE:' + str(
                    np.round(np.sqrt(mean_squared_error(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])), 2)))
                print('NAN ' + title[k] + ' R^2:' + str(
                    np.round(r2_score(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j]), 2)))

                print('NAN ' + title[k] + ' KS D:' + str(
                    np.round(ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)))
                print('NAN ' + title[k] + ' KS p:' + str(
                    np.round(ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)))

                plt.title(title[j], fontsize=20, fontweight='bold')
                ax.invert_yaxis()
                plt.xticks(fontsize=15, fontweight='bold')
                plt.yticks(testing_labels_locs, testing_labels,fontsize=15, fontweight='bold')

                matrix[j, k * 7]=np.round(np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1], 2) #pearson CC
                matrix[j, k * 7 + 1]=np.round(spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0],2) #spearman CC
                matrix[j, k * 7 + 2] = np.round(
                    spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)  # spearman p-val
                matrix[j, k * 7 + 3]=np.round(np.sqrt(mean_squared_error(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])), 2) #RMSE
                matrix[j, k * 7 + 4]=np.round(r2_score(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j]), 2) #r2
                matrix[j, k * 7 + 5] = np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)  # KS D
                matrix[j, k * 7 + 6] = np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)  # KS p val
                row_name[j][k]="NaN "+title[k]

                plt.tight_layout()
            elif j == 4:
                ax = plt.gca()
                plt.plot(org_x_test_data_new_mean[:, j], tim[:], color='k', label='Actual', linewidth=2)
                plt.plot(pred_x_test_data_new_mean[:, j], tim[:], color='r', label='Pred', linewidth=2)

                print('NAN ' + title[k] + ' Pearson CC:' + str(
                    np.round(np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1],
                             2)))  # perason
                print('NAN ' + title[k] + ' Spearman CC:' + str(
                    np.round(spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0],
                             2)))  # spearman
                print('NAN ' + title[k] + ' Spearman p-val:' + str(
                    np.round(spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1],
                             2)))  # spearman p-value
                print('NAN ' + title[k] + ' RMSE:' + str(
                    np.round(
                        np.sqrt(mean_squared_error(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])),
                        2)))
                print('NAN ' + title[k] + ' R^2:' + str(
                    np.round(r2_score(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j]), 2)))

                print('NAN ' + title[k] + ' KS D:' + str(
                    np.round(ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)))
                print('NAN ' + title[k] + ' KS p:' + str(
                    np.round(ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)))

                plt.title(title[j], fontsize=20, fontweight='bold')
                ax.invert_yaxis()
                plt.xticks(fontsize=15, fontweight='bold')
                plt.yticks(testing_labels_locs, testing_labels,fontsize=15, fontweight='bold')

                matrix[j, k * 7] = np.round(
                    np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1], 2)  # pearson CC
                matrix[j, k * 7 + 1] = np.round(
                    spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)  # spearman CC
                matrix[j, k * 7 + 2] = np.round(
                    spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)  # spearman p-val
                matrix[j, k * 7 + 3] = np.round(
                    np.sqrt(mean_squared_error(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])),
                    2)  # RMSE
                matrix[j, k * 7 + 4] = np.round(
                    r2_score(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j]), 2)  # r2
                matrix[j, k * 7 + 5] = np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)  # KS D
                matrix[j, k * 7 + 6] = np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)  # KS p val


                row_name[j][k] = "NaN DTS"

                plt.tight_layout()
            else:
                ax = plt.gca()

                print('NAN ' + title[k] + ' Pearson CC:' + str(
                    np.round(np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1],
                             2)))  # perason
                print('NAN ' + title[k] + ' Spearman CC:' + str(
                    np.round(spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0],
                             2)))  # spearman
                print('NAN ' + title[k] + ' Spearman p-val:' + str(
                    np.round(spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1],
                             2)))  # spearman p-value
                print('NAN ' + title[k] + ' RMSE:' + str(
                    np.round(
                        np.sqrt(mean_squared_error(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])),
                        2)))
                print('NAN ' + title[k] + ' R^2:' + str(
                    np.round(r2_score(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j]), 2)))

                print('NAN ' + title[k] + ' KS D:' + str(
                    np.round(ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)))
                print('NAN ' + title[k] + ' KS p:' + str(
                    np.round(ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)))

                plt.plot(org_x_test_data_new_mean[:, j], tim[:], color='k', label='Actual', linewidth=2)
                plt.plot(pred_x_test_data_new_mean[:, j], tim[:], color='b', label='Pred', linewidth=2)
                plt.title(title[j], fontsize=20, fontweight='bold')
                ax.invert_yaxis()
                plt.xticks(fontsize=15, fontweight='bold')
                plt.yticks(testing_labels_locs, testing_labels,fontsize=15, fontweight='bold')

                matrix[j, k * 7] = np.round(
                    np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1], 2)  # pearson CC
                matrix[j, k * 7 + 1] = np.round(
                    spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)  # spearman CC
                matrix[j, k * 7 + 2] = np.round(
                    spearmanr(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)  # spearman p-val
                matrix[j, k * 7 + 3] = np.round(
                    np.sqrt(mean_squared_error(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])),
                    2)  # RMSE
                matrix[j, k * 7 + 4] = np.round(
                    r2_score(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j]), 2)  # r2
                matrix[j, k * 7 + 5] = np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)  # KS D
                matrix[j, k * 7 + 6] = np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)  # KS p val

                row_name[j][k] = title[j]

                plt.tight_layout()
        #plt.show()
        print("=================================================================")

        np.save(save_path_model + 'testing_welllog_NaN_'+title[k]+'_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred.npy', pred_x_test_data_new_mean) #prediction of CVAE
        np.save(save_path_model + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred_lithology.npy', testing_seq_data_pred) #probability of lithology
        np.save(save_path_model + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred_result.npy', pred_x_test_data_result)

        np.save(save_path_model + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred_test_subtract.npy', pred_test_subtract_previous)
        np.save(save_path_model + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_testing_curve_CC_RMSE.npy',
                testing_nan_dts_data[0:m + 1, :])

        pred_test_subtract_previous = np.zeros((iteration, 1))
        k = k + 1

    result_pd = pd.DataFrame(matrix, columns=column_name)
    row_name_pd = pd.DataFrame(row_name)

    result_path = save_path_model+ 'testing_welllog_NaN_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred_result_excel.xlsx'

    with pd.ExcelWriter(result_path) as writer:
        result_pd.to_excel(writer, sheet_name='result')
        row_name_pd.to_excel(writer, sheet_name='info')
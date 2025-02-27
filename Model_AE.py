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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras import Model
from scipy.stats import spearmanr, ks_2samp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

### fix random seed
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

## parameterization
mode = 3
# mode=0: sequence data making and normalization #mode=1: AE model training #mode2=: AE model prediction #mode=3: AE missing data prediction
seq = 20
Nan_Number = 0
test_well_num = 0 # well 0: 1A, well 1: 1B, well 2: 11A, well 3: 11 T2, well 4: 14, well 5: 4
well_feature_num = 5
rock_class = 3
latent_dim = 2
window_size=seq

load_path = '../data/volve/well_6/medfilter_class3_0201_outlier/'
save_path_model = '../result/CVAE_LSTM_class3/save_compare_AE/seq'+str(seq)+'/'+str(test_well_num)+'/'
path_savefig = '../result/CVAE_LSTM_class3/figure_compare_AE/seq'+str(seq)+'/'+str(test_well_num)+'/'

def makedirs(path, path1):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path1)

makedirs(save_path_model, path_savefig)

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

def AE_Bayesian(batch_size, n_epoch, dense1, dense2, dense3):
    tf.random.set_seed(42)
    batch_size = int(batch_size)
    n_epoch = int(n_epoch)
    dense1 = int(dense1)
    dense2 = int(dense2)
    dense3 = int(dense3)

    ### Encoder ###
    encoder_inputs = Input(shape=(training_data_smote_res.shape[1]),name="encoder_inputs")
    dense1_layer = Dense(dense1, activation = 'relu',name="encoder_dense1")(encoder_inputs)
    encoder_dropout = Dropout(0.5)(dense1_layer)
    dense2_layer = Dense(dense2, activation='relu', name="encoder_dense2")(encoder_dropout)
    latent = Dense(dense3, activation='relu', name="encoder_dense3")(dense2_layer)
    encoder = keras.Model(inputs=encoder_inputs, outputs=latent)

    ### Decoder ###
    latent_inputs = keras.Input(shape=(latent.shape[1]))
    decoder_dense1 = Dense(dense2,activation='relu',name="decoder_dense1")(latent_inputs)
    decoder_dropout = Dropout(0.5)(decoder_dense1)
    decoder_dense2 = Dense(dense1, activation='relu', name="decoder_dense2")(decoder_dropout)
    decoder_out = Dense(encoder_inputs.shape[1], activation='sigmoid', name="decoder_output")(decoder_dense2)
    decoder = keras.Model(latent_inputs, decoder_out, name='decoder')

    class AE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.test_reconstruction_loss_tracker = keras.metrics.Mean(name='test_reconstruction_loss')
            self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')

        @property
        def metrics(self):
            return [
                self.reconstruction_loss_tracker,
                self.test_reconstruction_loss_tracker,
            ]

        def train_step(self, data):
            data=data[0]
            with tf.GradientTape() as tape:
                latent_data = self.encoder(data)
                reconstruction = self.decoder(latent_data)
                reconstruction_loss = tf.reduce_mean(keras.losses.MSE(data, reconstruction))

            grads = tape.gradient(reconstruction_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)

            return {
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            }

        def test_step(self, data):
            data = data[0]
            test_latent_data = self.encoder(data,training=False)
            test_reconstruction = self.decoder(test_latent_data, training=False)
            test_reconstruction_loss = tf.reduce_mean(keras.losses.MSE(data, test_reconstruction))

            self.test_reconstruction_loss_tracker.update_state(test_reconstruction_loss)
            return {"test_loss": self.test_reconstruction_loss_tracker.result()}

    AE_model = AE(encoder, decoder)
    AE_model.compile(optimizer=Adam())

    AE_model.fit(x=training_data_smote_res, y= training_data_smote_res, shuffle=True, epochs=n_epoch, batch_size=batch_size, verbose=0)

    score=AE_model.evaluate(x=validation_data, y= validation_data)

    return -1 * score

def AE_Bayesian_new(batch_size, n_epoch, dense1, dense2, dense3):
    tf.random.set_seed(42)

    batch_size = int(batch_size)
    n_epoch = int(n_epoch)
    dense1 = int(dense1)
    dense2 = int(dense2)
    dense3 = int(dense3)

    ### Encoder ###
    encoder_inputs = Input(shape=(training_data_smote_res.shape[1]), name="encoder_inputs")
    dense1_layer = Dense(dense1, activation='relu', name="encoder_dense1")(encoder_inputs)
    encoder_dropout = Dropout(0.5)(dense1_layer)
    dense2_layer = Dense(dense2, activation='relu', name="encoder_dense2")(encoder_dropout)
    latent = Dense(dense3, activation='relu', name="encoder_dense3")(dense2_layer)
    encoder = keras.Model(inputs=encoder_inputs, outputs=latent)

    ### Decoder ###
    latent_inputs = keras.Input(shape=(latent.shape[1]))
    decoder_dense1 = Dense(dense2, activation='relu', name="decoder_dense1")(latent_inputs)
    decoder_dropout = Dropout(0.5)(decoder_dense1)
    decoder_dense2 = Dense(dense1, activation='relu', name="decoder_dense2")(decoder_dropout)
    decoder_out = Dense(encoder_inputs.shape[1], activation='sigmoid', name="decoder_output")(decoder_dense2)
    decoder = keras.Model(latent_inputs, decoder_out, name='decoder')

    class AE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.test_reconstruction_loss_tracker = keras.metrics.Mean(name='test_reconstruction_loss')
            self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')

        @property
        def metrics(self):
            return [
                self.reconstruction_loss_tracker,
                self.test_reconstruction_loss_tracker,
            ]

        def train_step(self, data):
            data = data[0]
            with tf.GradientTape() as tape:
                latent_data = self.encoder(data)
                reconstruction = self.decoder(latent_data)
                reconstruction_loss = tf.reduce_mean(keras.losses.MSE(data, reconstruction))

            grads = tape.gradient(reconstruction_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            return {
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            }

        def test_step(self, data):
            data = data[0]
            test_latent_data = self.encoder(data, training=False)
            test_reconstruction = self.decoder(test_latent_data, training=False)
            test_reconstruction_loss = tf.reduce_mean(keras.losses.MSE(data, test_reconstruction))
            self.test_reconstruction_loss_tracker.update_state(test_reconstruction_loss)
            return {"test_loss": self.test_reconstruction_loss_tracker.result()}

    AE_model = AE(encoder, decoder)
    AE_model.compile(optimizer=Adam())

    path_model = join(path_savefig + 'AE_structure_Lithology_seq' + str(seq) + '_encoder.png')
    tf.keras.utils.plot_model(AE_model.encoder, show_shapes=True, to_file=path_model, expand_nested=True)  # AE model Structure plot

    path_model = join(path_savefig + 'AE_structure_Lithology_seq' + str(seq) + '_decoder.png')
    tf.keras.utils.plot_model(AE_model.decoder, show_shapes=True, to_file=path_model, expand_nested=True)  # AE model Structure plot

    path_model = join(path_savefig + 'AE_structure_Lithology_seq' + str(seq) + '_ae.png')
    tf.keras.utils.plot_model(AE_model, show_shapes=True, to_file=path_model, expand_nested=True)  # AE model Structure plot

    AE_model.fit(x=training_data_smote_res, y= training_data_smote_res, shuffle=True, epochs=n_epoch, batch_size=batch_size, verbose=0)

    score = AE_model.evaluate(x=validation_data, y=validation_data)

    model_path = save_path_model + 'Well num_' + str(test_well_num) + '_AE_bayesian_best_fit_seq' + str(
        seq) + '_encoder.h5'
    AE_model.encoder.save(model_path)

    model_path = save_path_model + 'Well num_' + str(test_well_num) + '_AE_bayesian_best_fit_seq' + str(
        seq) + '_decoder.h5'
    AE_model.decoder.save(model_path)

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

        j=0
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

        lithology_output=Lithology_val[seq:row-seq].to_numpy() #t time rock data
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

    ### lithology_data_concatenate(previous and t depth)
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
    plt.savefig(path_savefig + 'litho_percent_AE_oversampling_' + '.jpg', dpi=400, bbox_inches='tight',pad_inches=0.2)

    litho_num_1_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[1]
    litho_num_2_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[2]
    litho_num_3_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[3]

    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy', training_data_smote_res) #ae training_data_prev
    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_training_lithology_' + str(seq) + '_over.npy', training_lithology_nan_seq_over) #training_liothology_nan_sequence_prev

    smote_data_t = training_data_smote_res[:,seq,:]
    all_data_litho_con = np.concatenate((smote_data_t, training_lithology_nan_seq_over.reshape(-1, 1)), axis=1)
    all_data_pd = pd.DataFrame(all_data_litho_con, columns=['GR', 'NPHI', 'RHOB', 'DTC', 'DTS', 'Lithology'])
    all_data_pd['Lithology']=all_data_pd['Lithology'].replace([1],'claystone')
    all_data_pd['Lithology'] = all_data_pd['Lithology'].replace([2], 'sandstone')
    all_data_pd['Lithology'] = all_data_pd['Lithology'].replace([3], 'limestone')

    sns.set(font_scale=1.5)
    hue_order=['sandstone','limestone','claystone']
    f = sns.pairplot(all_data_pd, hue='Lithology', hue_order=hue_order)
    f.fig.subplots_adjust(top=.95)
    f.fig.suptitle("Training smote Well log data pairplot_smote", fontsize=20, fontweight='bold')
    plt.savefig(path_savefig + 'training_smote_data_pair_plot_smote.jpg', dpi=400, bbox_inches='tight', pad_inches=0.2)

elif mode == 1: #AE training
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    training_data_smote = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy')

    training_data_smote_res = training_data_smote.reshape(training_data_smote.shape[0], -1)
    validation_data = validation_data.reshape(validation_data.shape[0], -1)
    testing_data = testing_data.reshape(testing_data.shape[0], -1)

    max_dense1 = int(training_data_smote_res.shape[1])
    min_dense1 = int(max_dense1 * 0.5)

    max_dense2 = min_dense1
    min_dense2 = int(max_dense2 * 0.5)

    max_dense3 = min_dense2
    min_dense3 = int(max_dense3 * 0.5)

    pbounds = {
        'batch_size': (1000, 2000),
        'n_epoch': (50, 500),
        'dense1': (min_dense1, max_dense1),
        'dense2': (min_dense2, max_dense2),
        'dense3': (min_dense3, max_dense3),
    }

    acc = []
    lossdata = []
    b0 = BayesianOptimization(f=AE_Bayesian, pbounds=pbounds, verbose=2, random_state=1004)
    b0.maximize(init_points=5, n_iter=5)
    print('\n==============================================================================')
    print("Bayesian Optimization parameter")
    pp(b0.max)
    print('\n==============================================================================')

    with open( save_path_model+'Well num_'+str(test_well_num)+'_bayesian_result_AE_seq' + str(seq) + '.pkl', 'wb') as f:
        pickle.dump(b0.max, f)

    fit_AE = AE_Bayesian_new(
        batch_size=b0.max['params']['batch_size'],
        n_epoch=b0.max['params']['n_epoch'],
        dense1=b0.max['params']['dense1'],
        dense2=b0.max['params']['dense2'],
        dense3=b0.max['params']['dense3'],
    )

    print(fit_AE)

elif mode == 2:

    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    training_data_smote = np.load(
        save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy')
    training_data_smote_res = training_data_smote.reshape(training_data_smote.shape[0], -1)

    training_data = training_data.reshape(training_data.shape[0], -1)
    validation_data = validation_data.reshape(validation_data.shape[0], -1)
    testing_data = testing_data.reshape(testing_data.shape[0], -1)

    AE_encoder=keras.models.load_model(save_path_model + 'Well num_' + str(test_well_num) + '_AE_bayesian_best_fit_seq' + str(
        seq) + '_encoder.h5')

    AE_decoder = keras.models.load_model(
        save_path_model + 'Well num_' + str(test_well_num) + '_AE_bayesian_best_fit_seq' + str(
            seq) + '_decoder.h5')

    load_MinMaxScaler_scaler = pickle.load(open(save_path_model + 'MinMaxScaler_save_seq_' + str(seq) + '.pkl', 'rb'))

    #prediction ae using real data
    training_latent = AE_encoder.predict(training_data)
    pred_training = AE_decoder.predict(training_latent)

    validation_latent = AE_encoder.predict(validation_data)
    pred_validation = AE_decoder.predict(validation_latent)

    testing_latent = AE_encoder.predict(testing_data)
    pred_testing = AE_decoder.predict(testing_latent)

    training_smote_latent = AE_encoder.predict(training_data_smote_res)
    pred_training_smote = AE_decoder.predict(training_smote_latent)

    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_training_seq_' + str(seq) + '.npy',
            pred_training)
    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_validation_seq_' + str(seq) + '.npy',
            pred_validation)
    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_testing_seq_' + str(seq) + '.npy',
            pred_testing)

    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_pred_training_seq_over_' + str(seq) + '.npy',
            pred_training_smote)

    title = ['GR', 'NPHI', 'RHOB', 'DTC', 'DTS']

    training_data=training_data.reshape(-1, seq*2+1, well_feature_num)
    pred_training=pred_training.reshape(-1, seq*2+1, well_feature_num)

    # train data plot
    org_x_train_data_new = training_data[:,seq,:]
    pred_x_train_data_new = pred_training[:,seq,:]

    org_x_train_data_new = load_MinMaxScaler_scaler.inverse_transform(org_x_train_data_new)
    pred_x_train_data_new = load_MinMaxScaler_scaler.inverse_transform(pred_x_train_data_new)

    fig = plt.figure(figsize=(25, 30))
    fig.suptitle("Training", fontsize=40, fontweight='bold', position=(0.5, 1))
    tim = np.linspace(1, training_data.shape[0], training_data.shape[0])
    for i in range(0, 5):
        plt.subplot(1, 5, i + 1)
        ax = plt.gca()
        plt.plot(org_x_train_data_new[:, i], tim[:], color='k', label='Actual', linewidth=2)
        plt.plot(pred_x_train_data_new[:, i], tim[:], color='r', label='Pred', linewidth=2)
        ax.invert_yaxis()
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        plt.title(title[i], fontsize=20, fontweight='bold')

    fig.tight_layout()
    plt.savefig(path_savefig + 'prediction_AE_training_' + '.jpg', dpi=400, bbox_inches='tight',pad_inches=0.2)
    validation_data = validation_data.reshape(-1, seq * 2 + 1, well_feature_num)
    pred_validation = pred_validation.reshape(-1, seq * 2 + 1, well_feature_num)

# validation data plot
    org_x_valid_data_new = validation_data[:,seq,:]
    pred_x_valid_data_new = pred_validation[:,seq,:]
    org_x_valid_data_new = load_MinMaxScaler_scaler.inverse_transform(org_x_valid_data_new)
    pred_x_valid_data_new = load_MinMaxScaler_scaler.inverse_transform(pred_x_valid_data_new)

    fig = plt.figure(figsize=(25, 30))
    tim = np.linspace(1, validation_data.shape[0], validation_data.shape[0])
    fig.suptitle("Validation", fontsize=40, fontweight='bold', position=(0.5, 1))
    for i in range(0, 5):
        plt.subplot(1, 5, i + 1)
        ax = plt.gca()
        plt.plot(org_x_valid_data_new[:, i], tim[:], color='k', label='Actual', linewidth=2)
        plt.plot(pred_x_valid_data_new[:, i], tim[:], color='r', label='Pred', linewidth=2)
        ax.invert_yaxis()
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        plt.title(title[i], fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path_savefig + 'prediction_AE_validation_' + '.jpg', dpi=400, bbox_inches='tight',pad_inches=0.2)

    testing_data = testing_data.reshape(-1, seq * 2 + 1, well_feature_num)
    pred_testing = pred_testing.reshape(-1, seq * 2 + 1, well_feature_num)

    # test data plot
    org_x_test_data_new = testing_data[:,seq,:]
    pred_x_test_data_new = pred_testing[:,seq,:]

    org_x_test_data_new = load_MinMaxScaler_scaler.inverse_transform(org_x_test_data_new)
    pred_x_test_data_new = load_MinMaxScaler_scaler.inverse_transform(pred_x_test_data_new)

    fig = plt.figure(figsize=(25, 30))
    tim = np.linspace(1, testing_data.shape[0], testing_data.shape[0])
    plt.suptitle("Test", fontsize=40, fontweight='bold', position=(0.5, 1))
    for i in range(0, 5):
        plt.subplot(1, 5, i + 1)
        ax = plt.gca()
        plt.plot(org_x_test_data_new[:, i], tim[:], color='k', label='Actual', linewidth=2)
        plt.plot(pred_x_test_data_new[:, i], tim[:], color='r', label='Pred', linewidth=2)
        ax.invert_yaxis()
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        plt.title(title[i], fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path_savefig + 'prediction_AE_testing_' + '.jpg', dpi=400, bbox_inches='tight',pad_inches=0.2)


elif mode == 3:
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    ####AE Load####
    AE_encoder = keras.models.load_model(
        save_path_model + 'Well num_' + str(test_well_num) + '_AE_bayesian_best_fit_seq' + str(
            seq) + '_encoder.h5')

    AE_decoder = keras.models.load_model(
        save_path_model + 'Well num_' + str(test_well_num) + '_AE_bayesian_best_fit_seq' + str(
            seq) + '_decoder.h5')

    load_MinMaxScaler_scaler = pickle.load(open(save_path_model + 'MinMaxScaler_save_seq_' + str(seq) + '.pkl', 'rb'))
    title = ['GR', 'NPHI', 'RHOB', 'DTC', 'DTS']

    ###matrix
    column_name = ["Perason CC", "Sperman CC", "Spearman p-val", "RMSE", "R2", "KS D", "KS p-val"] * (well_feature_num-1)
    matrix = np.zeros((well_feature_num, 7 * (well_feature_num - 1)))

    row_name = [[0 for l in range(well_feature_num-1)] for j in range(well_feature_num)]

    k = 0
    pred_test_NaN = np.zeros((testing_data.shape[0], testing_data.shape[1], testing_data.shape[2]))

    for i in range(0, well_feature_num-1):
        x_test_data_NaN = copy.deepcopy(testing_data)
        x_test_data_NaN[:, :, i] = np.random.rand(x_test_data_NaN.shape[0],x_test_data_NaN.shape[1])
        x_test_data_NaN[:, :, 4] = np.random.rand(x_test_data_NaN.shape[0],x_test_data_NaN.shape[1]) #DTS

        x_test_data_NaN = x_test_data_NaN.reshape(x_test_data_NaN.shape[0],-1)
        latent_data = AE_encoder.predict(x_test_data_NaN)
        pred_test_NaN = AE_decoder.predict(latent_data)

        pred_x_test_data_new = pred_test_NaN.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2])

        org_x_test_data_new = testing_data
        org_x_test_data_new = org_x_test_data_new[:, seq,:]
        org_x_test_data_new = load_MinMaxScaler_scaler.inverse_transform(org_x_test_data_new)

        pred_x_test_data_new = pred_x_test_data_new[:, seq,:]

        ## Mean trend
        moving_averages_pred_nan = np.zeros(
            (pred_x_test_data_new.shape[0] - window_size + 1, pred_x_test_data_new.shape[1]))

        for s in range(pred_x_test_data_new.shape[1]):  # 각 열에 대하여
            moving_averages_pred_nan[:, s] = np.convolve(pred_x_test_data_new[:, s], np.ones(window_size) / window_size,
                                                         mode='valid')
        pred_x_test_data_new_mean = moving_averages_pred_nan

        def calculate_excluded_indices(W):
            if W % 2 == 1:
                excluded_front = excluded_back = (W - 1) // 2
            else:
                excluded_front = (W // 2) - 1
                excluded_back = W // 2

            return excluded_front, excluded_back

        front_idx, back_idx = calculate_excluded_indices(window_size)

        org_x_test_data_new_mean = org_x_test_data_new[front_idx:-back_idx, :]

        ####### missing data figure plot ######
        testing_labels = []
        testing_labels_locs = []
        depth_testing = testing_depth[front_idx:-back_idx].reshape(-1)
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

                print('NAN ' + title[k] + 'Pearson CC:' + str(
                    np.round(np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1], 2)))
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
                print('NAN ' + title[k] + ' KS D:' + str(np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)))
                print('NAN ' + title[k] + ' KS p:' + str(np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)))

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

                row_name[j][k]="NaN "+title[k]

                plt.tight_layout()
            elif j == 4:
                ax = plt.gca()
                plt.plot(org_x_test_data_new_mean[:, j], tim[:], color='k', label='Actual', linewidth=2)
                plt.plot(pred_x_test_data_new_mean[:, j], tim[:], color='r', label='Pred', linewidth=2)

                print('NAN ' + title[k] + 'Pearson CC:' + str(
                    np.round(np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1], 2)))
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
                print('NAN ' + title[k] + ' KS D:' + str(np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)))
                print('NAN ' + title[k] + ' KS p:' + str(np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)))

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
                print('NAN ' + title[k] + 'Pearson CC:' + str(
                    np.round(np.corrcoef(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0, 1], 2)))
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
                print('NAN ' + title[k] + ' KS D:' + str(np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[0], 2)))
                print('NAN ' + title[k] + ' KS p:' + str(np.round(
                    ks_2samp(org_x_test_data_new_mean[:, j], pred_x_test_data_new_mean[:, j])[1], 2)))

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
        print("=================================================================")

        path_savefig_testing_nan = path_savefig + 'testing_welllog_NaN_'+title[k]+'_Well num_' + str(
            test_well_num) + '_model_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_nan, dpi=400, bbox_inches='tight', pad_inches=0.2)

        # testing scatter plot
        fig = plt.figure(figsize=(50, 10))
        title_name = 'Test_NAN ' + title[k] + ' scatter plot'
        fig.suptitle(title_name, fontsize=40, fontweight='bold', position=(0.5, 1))

        ax1 = plt.subplot(1, 5, 1)
        ax1.scatter(org_x_test_data_new_mean[:, 0], pred_x_test_data_new_mean[:, 0], zorder=1, facecolors='none', edgecolors='k')

        lims = [np.min([org_x_test_data_new_mean[:, 0].min(), pred_x_test_data_new_mean[:, 0].min()]),
                np.max([org_x_test_data_new_mean[:, 0].max(), pred_x_test_data_new_mean[:, 0].max()])]
        ax1.plot(lims, lims, 'r-', alpha=0.75, zorder=2)

        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel('Actual',fontsize=15, fontweight='bold')
        ax1.set_ylabel('Predicted',fontsize=15, fontweight='bold')
        ax1.set_title('GR Scatter Plot with 1:1 Line',fontsize=25, fontweight='bold')

        ax2 = plt.subplot(1, 5, 2)
        ax2.scatter(org_x_test_data_new_mean[:, 1], pred_x_test_data_new_mean[:, 1], zorder=1, facecolors='none', edgecolors='k')

        lims = [np.min([org_x_test_data_new_mean[:, 1].min(), pred_x_test_data_new_mean[:, 1].min()]),
                np.max([org_x_test_data_new_mean[:, 1].max(), pred_x_test_data_new_mean[:, 1].max()])]
        ax2.plot(lims, lims, 'r-', alpha=0.75, zorder=2)

        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_xlabel('Actual',fontsize=15, fontweight='bold')
        ax2.set_ylabel('Predicted',fontsize=15, fontweight='bold')
        ax2.set_title('NPHI Scatter Plot with 1:1 Line',fontsize=25, fontweight='bold')

        ax3 = plt.subplot(1, 5, 3)
        ax3.scatter(org_x_test_data_new_mean[:, 2], pred_x_test_data_new_mean[:, 2], zorder=1, facecolors='none', edgecolors='k')

        lims = [np.min([org_x_test_data_new_mean[:, 2].min(), pred_x_test_data_new_mean[:, 2].min()]),
                np.max([org_x_test_data_new_mean[:, 2].max(), pred_x_test_data_new_mean[:, 2].max()])]
        ax3.plot(lims, lims, 'r-', alpha=0.75, zorder=2)

        ax3.set_xlim(lims)
        ax3.set_ylim(lims)
        ax3.set_xlabel('Actual',fontsize=15, fontweight='bold')
        ax3.set_ylabel('Predicted',fontsize=15, fontweight='bold')
        ax3.set_title('RHOB Scatter Plot with 1:1 Line',fontsize=25, fontweight='bold')

        ax4 = plt.subplot(1, 5, 4)
        ax4.scatter(org_x_test_data_new_mean[:, 3], pred_x_test_data_new_mean[:, 3], zorder=1, facecolors='none', edgecolors='k')

        lims = [np.min([org_x_test_data_new_mean[:, 3].min(), pred_x_test_data_new_mean[:, 3].min()]),
                np.max([org_x_test_data_new_mean[:, 3].max(), pred_x_test_data_new_mean[:, 3].max()])]
        ax4.plot(lims, lims, 'r-', alpha=0.75, zorder=2)

        ax4.set_xlim(lims)
        ax4.set_ylim(lims)
        ax4.set_xlabel('Actual',fontsize=15, fontweight='bold')
        ax4.set_ylabel('Predicted',fontsize=15, fontweight='bold')
        ax4.set_title('DTC Scatter Plot with 1:1 Line',fontsize=25, fontweight='bold')

        ax5 = plt.subplot(1, 5, 5)
        ax5.scatter(org_x_test_data_new_mean[:, 4], pred_x_test_data_new_mean[:, 4], zorder=1, facecolors='none', edgecolors='k')

        lims = [np.min([org_x_test_data_new_mean[:, 4].min(), pred_x_test_data_new_mean[:, 4].min()]),
                np.max([org_x_test_data_new_mean[:, 4].max(), pred_x_test_data_new_mean[:, 4].max()])]
        ax5.plot(lims, lims, 'r-', alpha=0.75, zorder=2)

        ax5.set_xlim(lims)
        ax5.set_ylim(lims)
        ax5.set_xlabel('Actual',fontsize=15, fontweight='bold')
        ax5.set_ylabel('Predicted',fontsize=15, fontweight='bold')
        ax5.set_title('DTS Scatter Plot with 1:1 Line',fontsize=25, fontweight='bold')

        plt.tight_layout()
        path_savefig_testing_nan_scatter = path_savefig + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_AE_model_seq_' + str(seq) + '_scatter.jpg'
        plt.savefig(path_savefig_testing_nan_scatter, dpi=400, bbox_inches='tight', pad_inches=0.2)

        fig = plt.figure(figsize=(25, 15))

        plt.hist(org_x_test_data_new_mean[:,k], bins=30, color='blue', alpha=0.3, edgecolor='black',
                 label='Actual')
        plt.hist(pred_x_test_data_new_mean[:,k], bins=30, color='red', alpha=0.3, edgecolor='black',
                 label='Predicted')

        plt.xlabel('Value', fontsize=15, fontweight='bold')
        plt.ylabel('Frequency', fontsize=15, fontweight='bold')
        plt.title('NaN ' + str(title[k]) + ' Histogram of Actual and Predicted', fontsize=30, fontweight='bold')
        plt.legend(fontsize=50, prop={'weight': 'bold'})
        path_savefig_testing_nan_histogram = path_savefig + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_AE_model_seq_' + str(seq) + '_histogram.jpg'
        plt.savefig(path_savefig_testing_nan_histogram, dpi=400, bbox_inches='tight', pad_inches=0.2)

        fig = plt.figure(figsize=(25, 15))

        plt.hist(org_x_test_data_new_mean[:, 4], bins=30, color='blue', alpha=0.3, edgecolor='black',
                 label='Actual')
        plt.hist(pred_x_test_data_new_mean[:, 4], bins=30, color='red', alpha=0.3, edgecolor='black',
                 label='Predicted')

        plt.xlabel('Value', fontsize=15, fontweight='bold')
        plt.ylabel('Frequency', fontsize=15, fontweight='bold')
        plt.title('NaN ' + str(title[k]) + ' DTS Histogram of Actual and Predicted', fontsize=30, fontweight='bold')
        plt.legend(fontsize=50, prop={'weight': 'bold'})
        path_savefig_testing_nan_histogram = path_savefig + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_AE_model_seq_' + str(seq) + '_DTS_histogram.jpg'
        plt.savefig(path_savefig_testing_nan_histogram, dpi=400, bbox_inches='tight', pad_inches=0.2)

        np.save(save_path_model + 'testing_welllog_NaN_'+title[k]+'_Well num_' + str(
            test_well_num) + '_model_seq_' + str(seq) + '_pred.npy', pred_x_test_data_new_mean)

        k = k + 1

    result_pd = pd.DataFrame(matrix, columns=column_name)
    row_name_pd = pd.DataFrame(row_name)

    result_path = save_path_model+ 'testing_welllog_NaN_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred_result_excel_mean.xlsx'

    with pd.ExcelWriter(result_path) as writer:
        result_pd.to_excel(writer, sheet_name='result')
        row_name_pd.to_excel(writer, sheet_name='info')
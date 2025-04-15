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
# mode=0: sequence data making and normalization #model=1: LSTM model training #mode=2: training prediction #mode 3: prediction missing data
seq = 20
Nan_Number = 0
test_well_num = 5 # well 0: 1A, well 1: 1B, well 2: 11A, well 3: 11 T2, well 4: 14, well 5: 4
well_feature_num = 5
rock_class = 3
window_size = seq

load_path = '../data/volve/well_6/medfilter_class3_0201_outlier/'
save_path_model = '../result/CVAE_LSTM_class3/save_compare_LSTM/seq'+str(seq)+'/'+str(test_well_num)+'/'
path_savefig = '../result/CVAE_LSTM_class3/figure_compare_LSTM/seq'+str(seq)+'/'+str(test_well_num)+'/'

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

    ### lithology_data_concatenate(previous and t depth) ###
    train_lithology_concatenate=np.concatenate((train_lithology_seq,train_lithology_nan_seq),axis=1)
    validation_lithology_concatenate=np.concatenate((validation_lithology_seq, validation_lithology_nan_seq),axis=1)
    test_lithology_concatenate=np.concatenate((test_data_lithology_seq, test_data_lithology.reshape(-1,1)),axis=1)

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
    plt.savefig(path_savefig + 'litho_percent_LSTM_oversampling_' + '.jpg', dpi=400, bbox_inches='tight',pad_inches=0.2)

    litho_num_1_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[1]
    litho_num_2_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[2]
    litho_num_3_over = collections.Counter(training_lithology_nan_seq_over.reshape(-1))[3]

    np.save(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy', training_data_smote_res)

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

elif mode == 1: #LSTM training
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    training_data_smote_res = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy') #real smote data

    input_data = training_data_smote_res[:,:seq,0:4]
    output_data = training_data_smote_res[:,seq,4]

    validation_input_data = validation_data[:,:seq,0:4]
    validation_output_data = validation_data[:,seq,4]

    testing_input_data = testing_data[:,:seq,0:4]
    testing_output_data = testing_data[:,seq,4]

    def LSTM_test(batch_size, n_epoch, n_hidden1, n_hidden2, dropout_rate):
        tf.random.set_seed(42)

        batch_size = int(batch_size)
        n_epoch = int(n_epoch)
        n_hidden1 = int(n_hidden1)
        n_hidden2 = int(n_hidden2)
        dropout_rate = float(dropout_rate)

        model = Sequential()
        model.add(LSTM(n_hidden1, input_shape=(seq, well_feature_num-1), return_sequences=False))
        model.add(Flatten())
        model.add(Dense(n_hidden2, activation='tanh'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='Adam',loss='mean_squared_error')
        model.fit(input_data, output_data, batch_size=batch_size, epochs=n_epoch, validation_data=(validation_input_data, validation_output_data),verbose=0)

        score=model.evaluate(validation_input_data, validation_output_data)

        return score * -1

    pbounds={
        'batch_size': (1000, 2000),
        'n_epoch': (1, 200),
        'n_hidden1': (10, 20),
        'n_hidden2': (9, 15),
        'dropout_rate': (0.1, 0.5),
    }

    b1=BayesianOptimization(f=LSTM_test, pbounds=pbounds, verbose=2, random_state=1004)
    b1.maximize(init_points=5, n_iter=5)
    print('\n==============================================================================')
    print("Bayesian Optimization parameter ")
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
        model.add(LSTM(n_hidden1, input_shape=(seq, well_feature_num - 1), return_sequences=False))
        model.add(Flatten())
        model.add(Dense(n_hidden2, activation='tanh'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='Adam', loss='mean_squared_error')
        model.summary()

        path_model = join(path_savefig + 'LSTM_structure_Lithology_seq' + str(seq) + '.png')
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=path_model)

        history_curve=model.fit(input_data, output_data, batch_size=batch_size, epochs=n_epoch,
                  validation_data=(validation_input_data, validation_output_data),verbose=0)

        fig, loss_ax = plt.subplots()
        loss_ax.plot(history_curve.history['loss'], 'b', label='train loss')
        loss_ax.plot(history_curve.history['val_loss'], 'g', label='validation loss')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper right')

        plt.savefig(path_savefig + 'LSTM_learning_curve.jpg', dpi=400, bbox_inches='tight', pad_inches=0.2)

        score = model.evaluate(validation_input_data, validation_output_data)

        model.save(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_seq_' + str(seq) + '.h5')

        train_score = model.evaluate(input_data, output_data)
        print(train_score)

        validation_score = model.evaluate(validation_input_data, validation_output_data)
        print(validation_score)

        testing_score = model.evaluate(testing_input_data, testing_output_data)
        print(testing_score)

        training_predictions = model.predict(input_data)
        np.save(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_training_result_seq_' + str(seq) + '.npy',
                training_predictions)

        validation_predictions = model.predict(validation_input_data)
        np.save(
            save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_validation_result_seq_' + str(seq) + '.npy',
            validation_predictions)

        testing_predictions = model.predict(testing_input_data)
        np.save(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_testing_result_seq_' + str(
            seq) + '.npy',
                testing_predictions)

        return score * -1

    fit_LSTM=LSTM_test_Optimize(
        batch_size=b1.max['params']['batch_size'],
        n_epoch=b1.max['params']['n_epoch'],
        n_hidden1=b1.max['params']['n_hidden1'],
        n_hidden2=b1.max['params']['n_hidden2'],
        dropout_rate=b1.max['params']['dropout_rate'])

    print(fit_LSTM)

elif mode == 2:
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    training_predictions = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_training_result_seq_' + str(seq) + '.npy')
    validation_predictions = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_validation_result_seq_' + str(seq) + '.npy')
    testing_predictions = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_testing_result_seq_' + str(
            seq) + '.npy')

    training_data_smote_res = np.load(save_path_model + 'Well num_' + str(test_well_num) + '_training_seq_' + str(seq) + '_over.npy')

    training_data_smote_res_t = training_data_smote_res[:,seq,4]
    validation_data_t = validation_data[:,seq,4]
    testing_data_t = testing_data[:,seq,4]

    load_MinMaxScaler_scaler = pickle.load(open(save_path_model + 'MinMaxScaler_save_seq_' + str(seq) + '.pkl', 'rb'))
    scale = load_MinMaxScaler_scaler.scale_[4]
    min_ = load_MinMaxScaler_scaler.min_[4]

    ### training_data ###
    title = ['GR', 'NPHI', 'RHOB', 'DTC', 'DTS']

    # train data plot
    org_x_train_data_new = training_data_smote_res_t
    pred_x_train_data_new = training_predictions

    org_x_train_data_new = org_x_train_data_new/scale + min_
    pred_x_train_data_new = pred_x_train_data_new/scale + min_

    fig = plt.figure(figsize=(15, 70))
    fig.suptitle("Training", fontsize=100, fontweight='bold', position=(0.5, 1))
    tim = np.linspace(1, org_x_train_data_new.shape[0], org_x_train_data_new.shape[0])
    ax = plt.gca()
    plt.plot(org_x_train_data_new, tim[:], color='k', label='Actual', linewidth=2)
    plt.plot(pred_x_train_data_new, tim[:], color='r', label='Pred', linewidth=2)
    ax.invert_yaxis()
    plt.xticks(fontsize=40, fontweight='bold')
    plt.yticks(fontsize=40, fontweight='bold')
    fig.tight_layout()
    plt.savefig(path_savefig + 'prediction_LSTM_training_' + '.jpg', dpi=400, bbox_inches='tight', pad_inches=0.2)

    org_x_valid_data_new = validation_data_t
    pred_x_valid_data_new = validation_predictions
    org_x_valid_data_new = org_x_valid_data_new/scale + min_
    pred_x_valid_data_new = pred_x_valid_data_new/scale + min_

    fig = plt.figure(figsize=(15, 70))
    tim = np.linspace(1, validation_data.shape[0], validation_data.shape[0])
    fig.suptitle("Validation", fontsize=100, fontweight='bold', position=(0.5, 1))
    ax = plt.gca()
    plt.plot(org_x_valid_data_new, tim[:], color='k', label='Actual', linewidth=2)
    plt.plot(pred_x_valid_data_new, tim[:], color='r', label='Pred', linewidth=2)
    ax.invert_yaxis()
    plt.xticks(fontsize=40, fontweight='bold')
    plt.yticks(fontsize=40, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path_savefig + 'prediction_LSTM_validation_' + '.jpg', dpi=400, bbox_inches='tight', pad_inches=0.2)

    # test data plot
    org_x_test_data_new = testing_data_t
    pred_x_test_data_new = testing_predictions
    org_x_test_data_new = org_x_test_data_new/scale + min_
    pred_x_test_data_new = pred_x_test_data_new/scale + min_

    fig = plt.figure(figsize=(15, 70))
    tim = np.linspace(1, testing_data.shape[0], testing_data.shape[0])
    plt.suptitle("Test", fontsize=100, fontweight='bold', position=(0.5, 1))
    ax = plt.gca()
    plt.plot(org_x_test_data_new, tim[:], color='k', label='Actual', linewidth=2)
    plt.plot(pred_x_test_data_new, tim[:], color='r', label='Pred', linewidth=2)
    ax.invert_yaxis()
    plt.xticks(fontsize=40, fontweight='bold')
    plt.yticks(fontsize=40, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path_savefig + 'prediction_LSTM_testing_' + '.jpg', dpi=400, bbox_inches='tight', pad_inches=0.2)

elif mode == 3:
    training_data, validation_data, testing_data, training_Lithology_new, validation_Lithology_new, testing_Lithology_new, training_Lithology_val_new, validation_Lithology_val_new, testing_Lithology_val_new, training_depth, validation_depth, testing_depth = load_data()

    ####LSTM Load####
    load_MinMaxScaler_scaler = pickle.load(open(save_path_model + 'MinMaxScaler_save_seq_' + str(seq) + '.pkl', 'rb'))
    data_min = load_MinMaxScaler_scaler.data_min_[4]
    data_max = load_MinMaxScaler_scaler.data_max_[4]

    model = tf.keras.models.load_model(
        save_path_model + 'Well num_' + str(test_well_num) + '_LSTM_model_seq_' + str(seq) + '.h5')

    testing_input_data = testing_data[:, : seq, 0:4]
    testing_output_data = testing_data[:, seq, 4]

    title = ['GR', 'NPHI', 'RHOB', 'DTC', 'DTS']

    ###matrix
    column_name = ["Perason CC", "Sperman CC", "Spearman p-val", "RMSE", "R2", "KS D", "KS p-val"] * (well_feature_num-1)
    matrix = np.zeros((1, 7 * (well_feature_num - 1)))

    row_name = [[0 for l in range(well_feature_num-1)] for j in range(1)]

    k = 0
    pred_test_NaN = np.zeros((testing_data.shape[0], 1))

    for i in range(0, well_feature_num-1):
        x_test_data_NaN = copy.deepcopy(testing_input_data)
        x_test_data_NaN[:, :, i] = np.random.rand(x_test_data_NaN.shape[0], x_test_data_NaN.shape[1])
        testing_seq_data_pred = model.predict(x_test_data_NaN)
        testing_seq_data_pred = testing_seq_data_pred.reshape(-1)

        moving_averages_pred_nan = np.zeros((testing_seq_data_pred.shape[0] - window_size + 1))
        moving_averages_pred_nan = np.convolve(testing_seq_data_pred, np.ones(window_size) / window_size,mode='valid')

        pred_x_test_data_new_mean = moving_averages_pred_nan

        print("=================================================================")
        testing_score = model.evaluate(x_test_data_NaN, testing_output_data)

        print(testing_score)
        print("=================================================================")

        def calculate_excluded_indices(W):
            if W % 2 == 1:
                excluded_front = excluded_back = (W - 1) // 2
            else:
                excluded_front = (W // 2) - 1
                excluded_back = W // 2

            return excluded_front, excluded_back
        front_idx, back_idx=calculate_excluded_indices(window_size)

        testing_labels = []
        testing_labels_locs = []
        depth_testing=testing_depth[front_idx:-back_idx].reshape(-1)
        for i in range(0, depth_testing.shape[0]):
            if (depth_testing[i] % 100) == 0:
                testing_labels.append(str(depth_testing[i]))
                testing_labels_locs.append(i)

        org_x_test_data_new = testing_output_data*(data_max-data_min) + data_min
        org_x_test_data_new = org_x_test_data_new[front_idx:-back_idx]

        pred_x_test_data_new = moving_averages_pred_nan*(data_max-data_min) + data_min
        pred_x_test_data_new = pred_x_test_data_new.reshape(-1)

        fig = plt.figure(figsize=(15, 70))
        tim = np.linspace(1, org_x_test_data_new.shape[0], org_x_test_data_new.shape[0])
        title_name = 'Test_NAN ' + title[k]
        fig.suptitle(title_name, fontsize=100, fontweight='bold', position=(0.5, 1))

        ax = plt.gca()
        plt.plot(org_x_test_data_new, tim[:], color='k', label='Actual', linewidth=2)
        plt.plot(pred_x_test_data_new, tim[:], color='r', label='Pred', linewidth=2)

        print('NAN ' + title[k] + ' Pearson CC:' + str(
            np.round(np.corrcoef(org_x_test_data_new, pred_x_test_data_new)[0, 1], 2)))
        print('NAN ' + title[k] + ' Spearman CC:' + str(
            np.round(spearmanr(org_x_test_data_new, pred_x_test_data_new)[0],
                     2)))  # spearman
        print('NAN ' + title[k] + ' Spearman p-val:' + str(
            np.round(spearmanr(org_x_test_data_new, pred_x_test_data_new)[1],
                     2)))  # spearman p-value
        print('NAN ' + title[k] + ' RMSE:' + str(
            np.round(np.sqrt(mean_squared_error(org_x_test_data_new, pred_x_test_data_new)), 2)))
        print('NAN ' + title[k] + ' R^2:' + str(
            np.round(r2_score(org_x_test_data_new, pred_x_test_data_new), 2)))
        print('NAN ' + title[k] + ' K-S D:' + str(
            np.round(
                ks_2samp(org_x_test_data_new, pred_x_test_data_new)[0], 2)))
        print('NAN ' + title[k] + ' K-S p:' + str(
            np.round(
                ks_2samp(org_x_test_data_new, pred_x_test_data_new)[1], 2)))

        ax.invert_yaxis()
        plt.xticks(fontsize=40, fontweight='bold')
        plt.yticks(testing_labels_locs, testing_labels,fontsize=40, fontweight='bold')

        matrix[0, k * 7] = np.round(np.corrcoef(org_x_test_data_new, pred_x_test_data_new)[0, 1],
                                    2)  # pearson CC
        matrix[0, k * 7 + 1] = np.round(spearmanr(org_x_test_data_new, pred_x_test_data_new)[0],
                                        2)  # spearman CC
        matrix[0, k * 7 + 2] = np.round(
            spearmanr(org_x_test_data_new, pred_x_test_data_new)[1], 2)  # spearman p-val
        matrix[0, k * 7 + 3] = np.round(np.sqrt(mean_squared_error(org_x_test_data_new, pred_x_test_data_new)), 2) # RMSE
        matrix[0, k * 7 + 4] = np.round(r2_score(org_x_test_data_new, pred_x_test_data_new),
                                        2)  # r2
        matrix[0, k * 7 + 5] = np.round(
            ks_2samp(org_x_test_data_new, pred_x_test_data_new)[0], 2)  # KS D

        matrix[0, k * 7 + 6] = np.round(
            ks_2samp(org_x_test_data_new, pred_x_test_data_new)[1], 2)  # KS p val

        row_name[0][k]="NaN "+title[k]
        plt.tight_layout()

        print("=================================================================")

        path_savefig_testing_nan = path_savefig + 'testing_welllog_NaN_'+title[k]+'_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '.jpg'
        plt.savefig(path_savefig_testing_nan, dpi=400, bbox_inches='tight', pad_inches=0.2)

        fig = plt.figure(figsize=(15, 15))
        ax = plt.gca()
        ax.scatter(org_x_test_data_new, pred_x_test_data_new, zorder=1, facecolors='none', edgecolors='k')
        lims = [np.min([org_x_test_data_new.min(), pred_x_test_data_new.min()]),
                np.max([org_x_test_data_new.max(), pred_x_test_data_new.max()])]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=2)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Actual',fontsize=15, fontweight='bold')
        ax.set_ylabel('Predicted',fontsize=15, fontweight='bold')
        ax.set_title('NaN '+str(title[k])+' Scatter Plot with 1:1 Line',fontsize=30, fontweight='bold')

        plt.tight_layout()
        path_savefig_testing_nan_scatter = path_savefig + 'testing_welllog_NaN_'+title[k]+'_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_scatter.jpg'
        plt.savefig(path_savefig_testing_nan_scatter, dpi=400, bbox_inches='tight', pad_inches=0.2)

        fig = plt.figure(figsize=(25,15))
        plt.hist(org_x_test_data_new,bins=30, color='blue', alpha=0.3, edgecolor='black', label='Actual')
        plt.hist(pred_x_test_data_new,bins=30, color='red', alpha=0.3, edgecolor='black', label='Predicted')
        plt.xlabel('Value',fontsize=15, fontweight='bold')
        plt.ylabel('Frequency',fontsize=15, fontweight='bold')
        plt.title('NaN '+str(title[k])+' Histogram of Actual and Predicted',fontsize=30, fontweight='bold')
        plt.legend(fontsize=50, prop={'weight': 'bold'})
        path_savefig_testing_nan_histogram = path_savefig + 'testing_welllog_NaN_' + title[k] + '_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_histogram.jpg'
        plt.savefig(path_savefig_testing_nan_histogram, dpi=400, bbox_inches='tight', pad_inches=0.2)
        np.save(save_path_model + 'testing_welllog_NaN_'+title[k]+'_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred.npy', pred_x_test_data_new)
        k = k + 1

    result_pd = pd.DataFrame(matrix, columns=column_name)
    row_name_pd = pd.DataFrame(row_name)

    result_path = save_path_model+ 'testing_welllog_NaN_Well num_' + str(
            test_well_num) + '_LSTM_model_seq_' + str(seq) + '_pred_result_excel.xlsx'

    with pd.ExcelWriter(result_path) as writer:
        result_pd.to_excel(writer, sheet_name='result')
        row_name_pd.to_excel(writer, sheet_name='info')
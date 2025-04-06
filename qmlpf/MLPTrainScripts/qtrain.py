## This file is part of https://github.com/SensorsINI/dnd_hls. 
## This intellectual property is licensed under the terms of the project license available at the root of the project.

# this script trains an =MLPF with dataset test data

# run this script from the MLPTrainScripts folder
# python qtrain.py 0 4
# ```
#  * first parameter 0/1, 0 means quantized training, 1 means float training.
#  * second parameter, 4 means quantized bits for inputs and activations before the last layer. the last layer have 16 bits. Ignored for training floating point network

# see README.md for more information

from __future__ import print_function
import os

## training/testing csv files here
if os.name=='nt':
    trainfilepath = 'G:/Downloads/2xTrainingDataDND21train'
    testfilepath = 'G:/Downloads/2xTrainingDataDND21test'
else:
    trainfilepath = '/home/tobi/Downloads/2xTrainingDataDND21train'
    testfilepath = '/home/tobi/Downloads/2xTrainingDataDND21test'

## network parameters here
num_hidden_layers=1 # use 1 for single hidden layer, 0 for perceptron
num_hidden_units = 20 # number of units in each hidden layer
actual_patch_size = 7 # size of input patches to MLP: actual_patch_size*actual_patch_size, should be odd integer. Can be up to 25x25 from default jAER NTF CSV output
# tau = 100000 # 300ms if use exp decay for preprocessing
tau_ms = 100  # time window for decay of time surface. Features to be filtered in should fit in the actual_patch_size window during this duration
encodemethod =  'timesurface' # 'cnn' # 'timesurface' # how to encode the timestamp input features, 'timesurface' for published float or quantized float time surface, 'bin' for binarized, 'lbb" for relative median method

SNR=1 #  1 for original balanced DND21 dataset. This *is* ratio of signal to noise events in dataset. 
epochs = 10 # TODO increase
learning_rate = 0.001
batch_size = 500 # training batch size
dropout=.5 # dropout allowed now in jAER with TF 1.5.0 by fixing toString() exception with try/catch in jAER in NTF commit 15.3.25

import datetime
from cmath import polar
import time

# from numpy.core.numeric import load
# np.random.seed(1337)
import os, sys

import argparse

import numpy as np
from tensorflow.keras import optimizers, initializers

# following ma
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import my_logger, yes_or_no

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# do not change these imports without knowing what you are doing.
# if you import keras from TF2, then the resulting MLP cannot be loaded into TF1 in jAER!!!

import tensorflow
from tensorflow import keras
from keras import layers, models
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.initializers import Orthogonal

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os, glob
import pandas as pd

# from sklearn.model_selection import train_test_split
# from custom_dataloader.dataloader import DataGenerator

from qkeras import *
from qkeras.utils import model_save_quantized_weights
from weighted_binary_cross_entropy import weighted_binary_crossentropy
from gabor_initializer_1d_patch import gabor_initializer_1d_patch

from convertH5toPB import freeze_session # to save pb model
log=my_logger(__name__)

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

MODEL_DIR = 'models' # where models and plots are saved
# DATASET_DIR='training_data'
# DATASET_DIR='g:/Downloads/particles/'
# DATASET_DIR='training_data/particles-real-noise-10-100pps'
# DATASET_DIR='training_data/particles-combined'
# dataset location
# trainfilepath = '../2xTrainingDataDND21train'
# testfilepath = '../2xTrainingDataDND21test'
# trainfilepath = os.path.join(DATASET_DIR,'particles_train')
# testfilepath = os.path.join(DATASET_DIR,'particles_test')
# trainfilepath = os.path.join(DATASET_DIR,'train')
# testfilepath = os.path.join(DATASET_DIR,'test')

TRAINING_SET_PATCH_SIZE = 25 # size of actual recorded patches from jAER NoiseTesterFilter
CSVINPUTLEN = TRAINING_SET_PATCH_SIZE * TRAINING_SET_PATCH_SIZE

global prefix




def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    classes = ['Not Fall', 'Fall']
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(os.path.join(MODEL_DIR,savename), format='png')
    plt.show()

#LossHistory, keep loss and acc
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))
 
    def loss_plot(self, model_dir, loss_type, e0loss, e0acc, e0valloss, e0valacc):

        iters = range(len(self.losses[loss_type])+1)
        # np.save(model_dir + 'loss.npy', np.array([e0loss] + self.losses[loss_type]))
        # np.save(model_dir + 'acc.npy', np.array([e0acc] + self.accuracy[loss_type]))

        plt.figure()
        # acc
        plt.plot(iters, [e0acc] + self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, [e0loss] + self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, [e0valacc] + self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, [e0valloss] + self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        
        plt.savefig(os.path.join(model_dir,'training.pdf'))

import math

def lbppreprocessingresize(allfeatures, resize, targetEventTS, targetEventP): # auc 0.85/0.80
    """ binarized by median relative timestamp, if older than median set to zero, if younger set to 1 """

    # print(features.shape)
    # print(features)
    # print(allfeatures.shape, targetEventTS.shape, targetEventP.shape)
    absTS = allfeatures[:, :CSVINPUTLEN]
    polarity = allfeatures[:, CSVINPUTLEN:]
    
    features = absTS#.transpose()

    mid = math.floor(len(features) / 2)
    featuresNormed = np.array([(features[i,:] >= np.median(features[i,mid])) for i in range(len(features))])
    # print(featuresdiff)
    featuresNormed = featuresNormed.astype(np.int)

    # crop
    features = featuresNormed.reshape(featuresNormed.shape[0], TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE)
    margin = int((TRAINING_SET_PATCH_SIZE - resize) / 2)
    cropend = TRAINING_SET_PATCH_SIZE - margin
    features = features[:,margin:cropend, margin:cropend]
    features = features.reshape(features.shape[0],resize * resize)

    polarity = polarity.reshape(polarity.shape[0], TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE)
    margin = int((TRAINING_SET_PATCH_SIZE - resize) / 2)
    cropend = TRAINING_SET_PATCH_SIZE - margin
    channelP = polarity[:,margin:cropend, margin:cropend]
    channelP = channelP.reshape(channelP.shape[0], resize * resize)
    channelP[features==0] = 0 # set the polarity to be 0 if the event is too old, which means the ts features are 0
    channelP[:,int(resize*resize/2)] = targetEventP # ensure the center location has the classified event's polarity

    features2 = np.hstack((features,channelP))
    # print(features2.shape)
    return features2


def binpreprocessingresize(allfeatures, resize, targetEventTS, targetEventP):
    """Binarized input patch
    """
    # print(features.shape)
    # print(features)
    # print(allfeatures.shape, targetEventTS.shape, targetEventP.shape)
    absTS = allfeatures[:, :CSVINPUTLEN]
    polarity = allfeatures[:, CSVINPUTLEN:]
    features = absTS#.transpose()
    featuresdiff = [features[i,:] - targetEventTS[i] for i in range(len(features))]
    # print(featuresdiff)
    featuresdiff = np.array(featuresdiff, dtype=int)
    featuresdiff = featuresdiff/1000
    featuresdiff = featuresdiff.astype(int)
    featuresNormed = np.abs(featuresdiff) < tau_ms # binarized time surface, 1 if event in NNb is younger than tau, otherwise zero
    featuresNormed = featuresNormed.astype(int)

    # crop
    features = featuresNormed.reshape(featuresNormed.shape[0], TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE)
    margin = int((TRAINING_SET_PATCH_SIZE - resize) / 2)
    cropend = TRAINING_SET_PATCH_SIZE - margin
    features = features[:,margin:cropend, margin:cropend]
    features = features.reshape(features.shape[0],resize * resize)

    polarity = polarity.reshape(polarity.shape[0], TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE)
    margin = int((TRAINING_SET_PATCH_SIZE - resize) / 2)
    cropend = TRAINING_SET_PATCH_SIZE - margin
    channelP = polarity[:,margin:cropend, margin:cropend]
    channelP = channelP.reshape(channelP.shape[0], resize * resize)
    channelP[features==0] = 0 # set the polarity to be 0 if the event is too old, which means the ts features are 0
    channelP[:,int(resize*resize/2)] = targetEventP # ensure the center location has the classified event's polarity
    
    features2 = np.hstack((features,channelP))
    # print(features2.shape)
    return features2

def cnnpreprocessingresize(allfeatures, resize, targetEventTS, targetEventP):
    # print(features.shape)
    # print(features)
    # print(allfeatures.shape, targetEventTS.shape, targetEventP.shape)
    absTS = allfeatures[:, :CSVINPUTLEN].reshape(-1, TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE)
    polarity = allfeatures[:, CSVINPUTLEN:].reshape(-1, TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE) # reshape to 2d batchsize*25x25

    dt_us = np.array([absTS[i,:] - targetEventTS[i] for i in range(len(absTS[:]))]) # compute dt for timesstamps relative to current event
    # print(featuresdiff)

    ts_normed = (tau_ms - np.abs(dt_us/1000.)) * (1.0 / float(tau_ms))
    ts_normed = np.clip(ts_normed, 0, 1)

    # crop
    margin = int((TRAINING_SET_PATCH_SIZE - resize) / 2)
    cropend = TRAINING_SET_PATCH_SIZE - margin
    ts_normed = ts_normed[:,margin:cropend, margin:cropend]

    channelP = polarity[:,margin:cropend, margin:cropend]
    channelP[ts_normed==0] = 0 # set the polarity to be 0 if the event is too old, which means the ts features are 0

    cnn_input = np.stack((ts_normed,channelP), axis=3)
    # log.debug(f'shape of CNN input: {features2.shape}')
    return cnn_input


def preprocessingresize(allfeatures, resize, targetEventTS, targetEventP): # resizes input patch to resize * resize
    # print(features.shape)
    # print(features)
    # print(allfeatures.shape, targetEventTS.shape, targetEventP.shape)
    absTS = allfeatures[:, :CSVINPUTLEN]
    polarity = allfeatures[:, CSVINPUTLEN:]
    
    features = absTS#.transpose()
    # print(features.shape, polarity.shape, targetEventTS.shape)
    # print(features[:10])
    # normalization
    # featuresdiff = pd.DataFrame(np.array(features)) - pd.DataFrame(np.array(targetEventTS)) 
    # print(featuresdiff.shape)
    featuresdiff = [features[i,:] - targetEventTS[i] for i in range(len(features))]
    # print(featuresdiff)

    
    featuresdiff = np.array(featuresdiff, dtype=int)
    featuresdiff = featuresdiff/1000
    featuresdiff = featuresdiff.astype(int)
    featuresNormed = (tau_ms - np.abs(featuresdiff)) * 1.0 / tau_ms
    
    
    featuresNormed = np.clip(featuresNormed, 0, 1)
    # print(np.unique(featuresNormed)[:10])
    # featuresNormed = np.exp(-np.abs(featuresdiff)/tau)
    
    # featuresNormed = featuresNormed.transpose()

    # crop
    features = featuresNormed.reshape(featuresNormed.shape[0], TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE)
    margin = int((TRAINING_SET_PATCH_SIZE - resize) / 2)
    cropend = TRAINING_SET_PATCH_SIZE - margin
    features = features[:,margin:cropend, margin:cropend]
    features = features.reshape(features.shape[0],resize * resize)

    polarity = polarity.reshape(polarity.shape[0], TRAINING_SET_PATCH_SIZE, TRAINING_SET_PATCH_SIZE)
    margin = int((TRAINING_SET_PATCH_SIZE - resize) / 2)
    cropend = TRAINING_SET_PATCH_SIZE - margin
    channelP = polarity[:,margin:cropend, margin:cropend]
    channelP = channelP.reshape(channelP.shape[0], resize * resize)
    channelP[features==0] = 0 # set the polarity to be 0 if the event is too old, which means the ts features are 0
    channelP[:,int(resize*resize/2)] = targetEventP # ensure the center location has the classified event's polarity
    

    features2 = np.hstack((features,channelP))
    # print(features2.shape)
    return features2

def preprocessing(features, targetEventTS): # assumes that MLP input is full logged CSV patch of 25x25
    middle = int(TRAINING_SET_PATCH_SIZE * TRAINING_SET_PATCH_SIZE / 2)
    features = features.transpose()
    # normalization
    featuresdiff = features - targetEventTS
    featuresNormed = (tau_ms - np.abs(featuresdiff)) * 1.0 / tau_ms
    featuresNormed = np.clip(featuresNormed, 0, 1)
    # featuresNormed = np.exp(-np.abs(featuresdiff)/tau)

    featuresNormed = featuresNormed.transpose()
    return featuresNormed

train = {}
val = {}

import random as random
def getgeneratorbatches(files): # TODO not used anymore
    # print('start generator')
    # while 1:
    if 1:
        # print('loop generator')
        sumtrainbatches = 0
        sumtestbatches = 0

        # random.shuffle(files)
        # print(files)
        for file_ in files:
            try:
                # print(file_)
                if 'concat' in file_:
                    encoding = "utf_8_sig"
                else:
                    encoding = "utf_8"
                df = pd.read_csv(file_, usecols=[0] + [i for i in range(3, 5 + CSVINPUTLEN * 2)], header=0, )
                # df.fillna(0)
                
                zero = len(df[df.iloc[:,2] == 0])
                # random.shuffle(df.values)
                # print('read file', file_, zero, len(df)-zero)

                # if mode == 1: #Train
                # traindf = df.iloc[:int(len(df) * 1)]
                traindf = df
                # else:
                    # df = df.iloc[int(len(df) * 0.8):]
                # testbatches = int(np.ceil((len(df)-len(traindf))/batch_size))

                trainbatches = int(np.ceil(len(traindf)/batch_size))
                sumtrainbatches += trainbatches
                # sumtestbatches += testbatches

                
            except EOFError:
                print('error' + file_)
        # print(sumtrainbatches,sumtestbatches)
        return sumtrainbatches   

def mygenerator(files,mode,encodemethod):
    """ Generate input patches in batches for the DNN
    :param files: list of (maybe compressed with xz) CSV files
    :param mode: not used
    :param encodemethod: how to process the time surface
    """
    log.debug('starting data generator')
    while 1:
        # print('loop generator')
        sumbatches = 0
        random.shuffle(files)
        # print(files)
        for file_ in files:
            try:
                # print(file_)
                if 'concat' in file_:
                    encoding = "utf_8_sig"
                else:
                    encoding = "utf_8"
                print(f' loading {file_}')
                try:
                    df = pd.read_csv(file_, usecols=[0] + [i for i in range(3, 5 + CSVINPUTLEN * 2)], header=0, comment='#')
                except Exception as e:
                    log.warning(f'got exception trying to read {file_}: {e}')
                    continue
                df.fillna(0)
                # random.seed(1)
                # zero = len(df[df.iloc[:,2] == 0])
                random.shuffle(df.values)


                batches = int(np.ceil(len(df)/batch_size))
                sumbatches += batches
                for i in range(0, batches):
                    m_data = df[i*batch_size:min(len(df),i*batch_size+batch_size)].values
                    # print(m_data.shape,m_data[:10])
                    m_data = m_data.astype('float')
                    # print(m_data.shape)
                    if TRAINING_SET_PATCH_SIZE > actual_patch_size:
                        # come here if the patch is smaller than the CSV recorded patch
                        # sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*2], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
                        y = m_data[:,2] # this is the ground truth target class (0=noise 1=signal)
                        
                        # print(y.shape)
                        if encodemethod == 'bin': # fixed point input patche processing
                            x = binpreprocessingresize(m_data[:,3:3 + CSVINPUTLEN * 2], actual_patch_size, m_data[:, 1], m_data[:, 0])
                        elif encodemethod == 'lbp':
                            x = lbppreprocessingresize(m_data[:,3:3 + CSVINPUTLEN * 2], actual_patch_size, m_data[:, 1], m_data[:, 0])
                        elif encodemethod=='cnn':
                            x = cnnpreprocessingresize(m_data[:,3:3 + CSVINPUTLEN * 2], actual_patch_size, m_data[:, 1], m_data[:, 0])
                        elif encodemethod=='timesurface': # floating point input patch time surface preprocessing
                            x = preprocessingresize(m_data[:,3:3 + CSVINPUTLEN * 2], actual_patch_size, m_data[:, 1], m_data[:, 0])
                        else:
                            raise ValueError(f'unknown time surface encoding mode {encodemethod}')

                        # print(y.shape,y[:10])
                        # print(x.shape,x[:10])
                    else:
                        # sample = {'y': m_data[:,4], 'x': preprocessing(m_data[:,5:5+csvinputlen*2], m_data[:,3])}
                        y = m_data[:,4]
                        x = preprocessing(m_data[:,5:5 + CSVINPUTLEN * 2], m_data[:, 3])
                    # if self.transform:
                    #     sample = self.transform(m_data)
                    
                    # print(x[:10])
                    # print(y[:10])
                    # exit(0)
                    yield(x,y)
            except EOFError:
                print('error' + file_)
        # print(f'{sumbatches} batches done')


def splittraintest(csvdir, splitratio): # TODO not used now
    
    allFiles = glob.glob(os.path.join(csvdir,'*TI25*.csv'))
    if len(allFiles) > 0:
        np_array_list = []
        for file_ in allFiles:
            print(file_)
            df = pd.read_csv(file_, usecols=[0] + [i for i in range(3, 5 + CSVINPUTLEN * 2)], header=0) # might change if the collecting code change
            np_array_list.append(df.values)

        # read all csv files in a folder to one wpndas frame
        # comb_np_array = np.vstack(np_array_list)
        # np.random.shuffle(comb_np_array)
        # big_frame = pd.DataFrame(comb_np_array)
        # # big_frame = big_frame.fillna(0)
        # print(big_frame.head(), big_frame.size)

        # # split train and test set
        # leng = len(big_frame)
        # trainset = big_frame.iloc[:int(splitratio*leng)]
        # testset = big_frame.iloc[int(splitratio*leng):]
        # del big_frame 
        # del msk
        # trainlist = np_array_list[:int(splitratio*)]
        return pd.DataFrame(np.vstack(np_array_list[:int(splitratio*len(np_array_list))])), pd.DataFrame(np.vstack(np_array_list[int(splitratio*len(np_array_list)):]))


def qbuildDModel(resize, hidden):
    networkinputlen = resize * resize
    if resize > 0:
        inputs = Input(shape=[networkinputlen*2, ], name='input')
        x = Dense(hidden, input_shape=(networkinputlen*2, ), activation='relu', name='fc1')(inputs)
        x = Dense(hidden, activation='relu', name='fc2')(x)
        # x = Dropout(0.2)(x)
        x = Dense(1, activation='sigmoid', name='output')(x)
        model = Model(inputs, x)

        return model

def build_mlp_model(patch_size, num_hidden_units, num_hidden_layers=1, output_bias=None, dropout=0.5): # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
#     if output_bias is not None:
#         output_bias = tf.keras.initializers.Constant(output_bias)
#     networkinputlen = patch_size * patch_size
#     if patch_size > 0:
#         gabor_init = gabor_initializer_1d_patch(
#             patch_height=patch_size,
#             patch_width=patch_size,
#             num_filters=num_hidden_units
#         )
#         inputs = Input(shape=[2*networkinputlen, ], name='input')
#         # initializer=Orthogonal(shape=(patch_size,patch_size))
#         x = Dense(num_hidden_units, input_shape=(networkinputlen * 2,), activation='relu', name='fc1', kernel_initializer=gabor_init)(inputs)
#         if dropout>0:
#             x = Dropout(dropout)(x)

#         for i in range(num_hidden_layers-1):
#             x = Dense(num_hidden_units, 
#                       input_shape=(num_hidden_units,), 
#                       activation='relu', 
#                       name=f'fc{i+2}',
#                       kernel_initializer=Orthogonal(),
# )(x)
#             if dropout > 0:
#                 x = Dropout(dropout)(x)

#         x = Dense(1, activation='sigmoid', name='output',
#                          bias_initializer=output_bias)(x)
        
#         model = Model(inputs, x)
        from mymlp import MyMLP
        model = MyMLP(patch_size=patch_size,num_hidden_layers=1,num_hidden_units=num_hidden_units, dropout=dropout)
        
        return model

def build_cnn_model(patch_size, output_bias=None, dropout=0.5): # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(patch_size,patch_size, 2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu',bias_initializer=output_bias))
    model.add(layers.Dense(1, activation='sigmoid',bias_initializer=output_bias))
    return model


def qbuildModel(resize, hidden, bits):
    networkinputlen = resize * resize
    
    if resize > 0:
        inputs = Input(shape=[networkinputlen*2, ], name='input')
        x = QActivation("quantized_bits(" +str(bits)+ ")", name="qact0")(inputs)
        x = QDense(hidden, input_shape=(networkinputlen*2, ), kernel_quantizer=quantized_bits(bits),
           bias_quantizer=quantized_bits(bits),
           name="fc1")(x)  
        x = QActivation("quantized_relu(" +str(bits)+ ")", name="relu1")(x)
        x = QDense(1, kernel_quantizer=quantized_bits(bits),
           bias_quantizer=quantized_bits(bits),
           name="fc2")(x)
        x = QActivation("quantized_relu(" +str(16)+ ")", name="doutput")(x)
        x = QActivation("quantized_relu(bits=16, integer=0, use_sigmoid=1)",name='output')(x)
        model = Model(inputs, x)

        return model

# def qbuildModel(resize, hidden):
#     networkinputlen = resize * resize
#     if resize > 0:
#         inputs = Input(shape=[networkinputlen*2, ], name='input')
#         x = QActivation("quantized_bits(8)", name="qact0")(inputs)
#         x = QDense(hidden, input_shape=(networkinputlen*2, ), kernel_quantizer=quantized_bits(8),
#            bias_quantizer=quantized_bits(8),
#            name="fc1")(x)  
#         # x = QDense(hidden, input_shape=(networkinputlen*2, ), kernel_quantizer=quantized_bits(8),
#         #    bias_quantizer=quantized_bits(8),
#         #    name="fc1")(inputs)  
#         x = QActivation("quantized_relu(9)", name="relu1")(x)

#         # x = Dense(1, activation='sigmoid', name='output')(x)
#         x = QDense(1, kernel_quantizer=quantized_bits(8),
#            bias_quantizer=quantized_bits(8),
#            name="fc2")(x)
#         x = Activation("sigmoid", name="output")(x)
#         model = Model(inputs, x)

#         return model


def qbuildPerceptron(resize):
    networkinputlen = resize * resize
    if resize > 0:
        inputs = Input(shape=[networkinputlen*2, ], name='input')
        x = Dense(1, input_shape=(networkinputlen*2, ),  activation='sigmoid', name='output')(inputs)
        model = Model(inputs, x)

        return model


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y_true,y_score,model_dir):
    fpr,tpr,threshold = roc_curve(y_true,y_score,pos_label=1)
    auc = roc_auc_score(y_true,y_score)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('roc curve' + str(auc))
    plt.plot(fpr,tpr,color='b',linewidth=1)
    plt.plot([0,1],[0,1],'r--')
    plt.savefig(os.path.join(model_dir , 'roccurve.pdf'))
    plt.clf()


# trainbatches = 2705#3842
# testbatches = 784#2078#938
def trainFunction(trainfiles, testfiles, trainbatches, testbatches, patch_size, hidden, epochs, repeat, mtype, Train, bits, dropout=0.0, num_hidden_layers=1): # todo repeat same as num_hidden_layers
    pos = 1/(1+1/SNR)  # count of positive class, i.e. if SNR=1, equal # signal and noise, then signal is half of total
    neg = 1/(1+SNR)  # count of negative class
    total = pos + neg
    weight_for_0 = (1+SNR)/2
    weight_for_1 = (1+1/SNR)/2 # so that SNR=1 produces 1:1 weights
    print(f'binary cross entropy weights for SNR={SNR:.3f} are signal:{weight_for_1:.3f} noise:{weight_for_0:.3f}')
    class_weight = {0: weight_for_0, 1: weight_for_1}
    initial_bias = np.log(SNR)  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    loss_fn = weighted_binary_crossentropy(pos_weight=weight_for_1)

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        # keras.metrics.Precision(name='precision'),
        # keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc',num_thresholds=2000),
        # keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]

    prefix = ''
    log.info(f'training {mtype} model')
    model=None
    if mtype == 'double': # two hidden layer floating-point MLP
        model = qbuildDModel(patch_size, hidden)
        middlefix = 'DH'
    elif mtype == 'qsingle': # quanitized MLP with one hidden layer
        model = qbuildModel(patch_size, hidden, bits)
        middlefix = 'qH'
        prefix = '16bit-qsigmoidput' + encodemethod + str(tau_ms) + 'tau' + str(bits) + 'bitaw' + str(repeat) + 'MSEO1' + middlefix + str(hidden) + '_linear_' + str(patch_size)
    elif mtype == 'mlp': # floating MLP with one hidden layer
        model = build_mlp_model(patch_size=patch_size, num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers, output_bias=initial_bias, dropout=dropout)
        middlefix = 'fH'
        prefix = 'float' + encodemethod + str(tau_ms) + 'tau' + str(bits) + 'bitaw' + str(repeat) +  middlefix + str(hidden) + '_linear_' + str(patch_size)

    elif mtype == 'cnn': # floating MLP with one hidden layer
        model = build_cnn_model(patch_size=patch_size, output_bias=initial_bias, dropout=dropout)
        middlefix = 'cnn'
        prefix = 'float' + encodemethod + str(tau_ms) + 'tau' +  middlefix  + '_linear_' + str(patch_size)
    elif mtype == 'perceptron': # single layer pereception (no hidden layer)
        model = qbuildPerceptron(patch_size)
        middlefix = 'H'
    else:
        raise TypeError(f'model type {mtype} is not recognized')

    from pathlib import Path
    Path(MODEL_DIR).mkdir(exist_ok=True)
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_dir_name = os.path.join(MODEL_DIR, prefix + '-' + datestr)
    Path(model_dir_name).mkdir(exist_ok=True)
    
    log.info(f'will save model to files in {model_dir_name}')
    
    if mtype=='mlp':
        from plotmlpfkernels import plot_kernels
        plot_kernels(model, model_dir_name, 'kernels-initial.png')

    # Path(model_name_base).mkdir(exist_ok=True)
    # print(f'made folder {model_name_base} to save model data to')

    if Train:
        traingenerator = mygenerator(trainfiles,1,encodemethod)
        # testx,testy = getxandy(testfiles[0])
        testgenerator = mygenerator(testfiles,0,encodemethod)
        
        log.info(f'building model...{prefix}')
        optimizer = Adam(learning_rate=learning_rate)
        # model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy']) # original loss used in DND21 paper with balanced signal and noise event counts
        # https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow


        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=METRICS
        )
        model.summary()


        # n_batches = testgenerator.samples//batch_size
        e0loss, e0acc = 0.25,0.49#model.evaluate_generator(traingenerator,steps=trainbatches,verbose=1) 
        e0valloss, e0valacc = 0.26,0.29#model.evaluate_generator(testgenerator, steps=testbatches, verbose=1) 
        print('No.%d newly-initialized net %s loss: %.2f, acc: %.2f'%(repeat,prefix,e0valloss,e0valacc))

        # model = qbuildModel(resize,hidden)

        history = LossHistory()

        filepath=os.path.join(MODEL_DIR,prefix + "-{epoch:02d}-{val_accuracy:.3f}.h5")
        checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',mode='max' ,save_best_only='True', save_weights_only='True', save_freq='epoch')
        # print(f'checkpoint={checkpoint}')

        # checkpoint = ModelCheckpoint(filepath='mlpmodels',monitor='loss',mode='auto' ,save_best_only='True')

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', # stop when validation AUC drops from overfitting
            verbose=1,
            patience=4,
            mode='max',
            restore_best_weights=True)

        model.fit_generator(
            generator=traingenerator,
            steps_per_epoch=trainbatches,
            epochs=epochs,
            validation_data=testgenerator,
            validation_steps=testbatches,
            callbacks=[history,checkpoint,early_stopping],
            verbose=1, workers=1, use_multiprocessing=False
        )

        history.loss_plot(model_dir_name,'epoch', e0loss, e0acc, e0valloss, e0valacc)

        model.export(model_dir_name)
        model_file_name = model_dir_name+'-model.h5'
        model.save(os.path.join(model_dir_name,model_file_name)) # export h5 file for converting to pb (but maybe loading from export would be better for making pb)
        # quantized_weights_filename=model_name_base+'weights.h5'
        # model_save_quantized_weights(model, quantized_weights_filename)

        plot_kernels(model, model_dir_name, 'kernels-final.png')

        all_weights = []
        for layer in model.layers:
            for w, weights in enumerate(layer.get_weights()):
                print(layer.name, w)
                all_weights.append(weights.flatten())

        all_weights = np.concatenate(all_weights).astype(np.float32)
        print(f'total number of weights: {all_weights.size}')

        print('layers:')
        for layer in model.layers:
          for w, weight in enumerate(layer.get_weights()):
            print(layer.name, w, weight.shape)

        # print_qstats(model)
        try:
            freeze_session(model_path=None,model=model) # use existing model object to freeze it to pb
            print(f'saved model as {model_file_name}')
        except Exception as e:
            print(f'could not convert h5 model to pb model; got: {e}')
        

    else: # test model TODO update below code
        print('testing saved model on data')
        # model = load_model( prefix + '.h5')
        # model.summary()
        # model = qbuildModel(resize,hidden)
        # model = load_model( prefix + '.h5')
        # model.summary()
        # model.compile(optimizer, loss)
        # model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])

        testgenerator = mygenerator(testfiles,0)
        # model = load_model( prefix + '.h5')
        # model.summary()
        # model = qbuildModel(resize,hidden)
        # model = load_model( prefix + '.h5')
        # model = load_model('0308pm16bitsqsigmoidput' + encodemethod + 'tau'+str(tau)+'quantize8awq2x8bitawawI7qH20-01-0.952.h5')
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # model.load_weights(prefix + 'wk.h5')
        # weightpath = '0308pm16bitsqsigmoidput' + encodemethod + 'tau'+str(tau)+'quantize8awq2x8bitawawI7qH20-02-0.817.h5'
        # weightpath = '0308pm16bitsqsigmoidput' + encodemethod + 'tau'+str(tau)+'quantize8awq2x8bitawawI7qH20-01-0.962.h5'
        # weightpath = '0308pm16bitsqsigmoidput' + encodemethod + 'tau'+str(tau)+'quantize8awq2x8bitaw0MSEO1qH20_linear_7wk.h5'
        weightpath = prefix + 'wk.h5'
        model.load_weights(weightpath)
        # model.summary()


    # return

    # n_batches = len(testgenerator)
    if True:
        print('generating training plots and ROC curve plot')
        # loss, acc = model.evaluate_generator(testgenerator,steps=testbatches, verbose=1) 
        # print('No.%d trained net %s loss: %.2f, acc: %.2f'%(repeat,prefix,loss,acc))
        # y_true = np.concatenate([testgenerator[i][1] for i in range(testbatches)])
        # initpredictions = model.predict_generator(testgenerator, steps=testbatches, verbose=1) 
        y_true = np.array([])
        initpredictions = np.array([])

        icount = 0
        for i,(batchx,batchy) in enumerate(testgenerator):
            
            if icount == testbatches:
                print('predict all batch', i)
                break
            icount +=1
            # print('predict batch', i)
            y_true = np.concatenate([y_true,batchy])
            # print(y_true.shape)
            # print(batchx.shape,batchy.shape,model.predict(batchx,verbose=0).shape)
            # return
            initpredictions = np.concatenate([initpredictions, np.squeeze(model.predict(batchx,verbose=0))])
            # print(initpredictions.shape)
        
        rocauc = roc_auc_score(y_true,initpredictions)
        print(prefix,'auc',rocauc)

        plot_roc_curve(y_true,initpredictions,model_dir_name)
        # del model
        # return
        y_pred = (initpredictions > 0.5).astype(int)
        y_true = np.reshape(y_true, [-1])
        y_pred = np.reshape(y_pred, [-1])
        print(y_true.shape, y_pred.shape)

        finderrorloc = y_true == y_pred
        indices = np.where(finderrorloc == False)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        TPR = recall_score(y_true, y_pred, average='binary')
        f1score = f1_score(y_true, y_pred, average='binary')

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn * 1.0 / (tn + fp) # TNR
        FPR = fp * 1.0 / (tn + fp) 

        print( prefix + '.h5')
        print('testacc:',accuracy)
        print('precision:',precision)
        print('FPR:',FPR)
        print('TPR:',TPR)
        print('f1score:',f1score)
        cm= confusion_matrix(y_true, y_pred)

        print('confusion matrix')
        print(cm)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('normalized confusion matrix')
        print(cm_normalized)

        # plot output distribution
        # initpredictions = np.array(initpredictions)
        # initpredictions = initpredictions.reshape((initpredictions.shape[0] * initpredictions.shape[1]))
        
        plt.clf()
        if Train:
            histlabel = 'network output'
        else:
            histlabel = 'newly-initialized network output'
        plt.hist(initpredictions, bins=20, label=histlabel)
        plt.xlim(0,1)
        plt.legend()
        plt.savefig(os.path.join(model_dir_name, 'outputhist.pdf'))


if __name__=='__main__':
    # main
    parser=argparse.ArgumentParser(description='trains MLPs for denoising DVS')
    # parser.add_argument('--train', action='store_true', help='train a model')
    parser.add_argument('--quantized', type=int, help='train quantized network with QUANTIZED bits of weight and state precision')
    parser.add_argument('--arch', choices=['perceptron','mlp','cnn'], help='type of network to train', default='mlp')

    if actual_patch_size%2==0:
        raise ValueError(f'actual_patch_size ({actual_patch_size}) should be odd')
    elif actual_patch_size>TRAINING_SET_PATCH_SIZE:
        raise ValueError(f'maximum actual_patch_size ({actual_patch_size}) is {TRAINING_SET_PATCH_SIZE}')


    args=parser.parse_args()
    if args.quantized:
        if args.bits is None:
            log.error('--bits N is missing for --quantized')
            quit(1)
            
    # training params
    # set SNR according to dataset signal and noise event counts, reported at end of recording CSV file from NTF in jAER
    # e.g. INFO: closed CSV output file drTraining.csv with 2,945,421 events (1,372,048 signal events, 1,573,373 noise events, SNR=0.872
    # java code is float snr = (float) csvSignalCount / (float) csvNoiseCount;
    # this number is used to set the binary cross entropy weighting so that at discrimation threshold 0.5, the accuracy of signal and noise classification is balanced(?)
    SNR=1 #  1 for original balanced DND21 dataset. This *is* ratio of signal to noise events in dataset. 
    # SNR= 0.0761 # particles-slow-real-noise
    # SNR= 0.465 # particles-real-noise-10-100pps
    # SNR= (0.0761+.465)/2 # particles-slow-real-noise
    # SNR= 1. / 100 # ratio of signal events to noise events in dataset. Set to 0.5 for original balanced DND21 paper; set to other values for e.g. high noise rate and low signal rate; see model.compile below
    # you can get this ratio at end of output of CSV file from jAER NoiseTesterFilter; See ReadMe.md
    print(f'\n*********\ntraining model with architecture {args.arch}\n'
        f'hidden layers: {num_hidden_layers}\n'
        f'hidden units per layer: {num_hidden_units} \n'
        f'input patch size: {actual_patch_size}x{actual_patch_size}\n'
        f'time surface encoding method: {encodemethod}\n'
        f'assuming tau_ms={tau_ms}\n'
        f'using training data from {trainfilepath}\n'
        f'using batch size {batch_size}, dropout={dropout}, learning rate {learning_rate}, and maximum of {epochs} epochs,\n'
        f'    and class ratio signal to noise SNR {SNR:.3f}'
        )

    if os.name!='nt':
        ok=yes_or_no('Are these parameters what you intend?', timeout=2) # timeout does not work on windows, user nust confirm
        if not ok:
            quit(0)
    else:
        time.sleep(5)
    
    start_time=time.time()

    from pathlib import Path
    trainfiles = glob.glob(str(Path(trainfilepath).joinpath('*.csv*'))) # trailing wildcard to read compressed csv files, e.g. xxx.csv.xz or xxx.csv
    testfiles = glob.glob(str(Path(testfilepath).joinpath('*.csv*'))) # trailing wildcard to read compressed csv files, e.g. xxx.csv.xz or xxx.csv
    if len(trainfiles)==0 or len(testfiles)==0:
        print(f'********there are no files in {trainfilepath} or in {testfilepath}; check trainfilepath and testfilepath variables and working folder')
        quit(1)
    else:
        print(f'** found {len(trainfiles)} training CSV files and {len(testfiles)} testing CSV files')

    # todo enlarge to really run
    trainbatches = batch_size #getgeneratorbatches(trainfiles) # todo very slow, replace with constants when running repeatedly
    testbatches = batch_size/10 #getgeneratorbatches(testfiles)
    print(f'batch size used for training={trainbatches}, for testing={testbatches}')

    trainflag=True # set to false for model testing

    trainFunction(trainfiles, testfiles, trainbatches, testbatches, actual_patch_size, num_hidden_units, epochs, 0, args.arch, trainflag, args.quantized, num_hidden_layers=num_hidden_layers, dropout=dropout)

    end_time=time.time()
    print(f'total elapsed time {end_time-start_time:.1f} seconds')
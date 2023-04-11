from __future__ import print_function
from operator import mod
from keras import callbacks
import numpy as np
from numpy.core.numeric import load
np.random.seed(1337)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.preprocessing import sequence
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense,Embedding,Dropout
from keras.layers import LSTM
from keras.datasets import imdb
from keras.callbacks import Callback
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score, confusion_matrix
import tensorflow as tf
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import os,sys,glob
import pandas as pd

from sklearn.model_selection import train_test_split
from keras_dataloader.dataloader import DataGenerator

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import scipy.io as sio

# training paras
learning_rate = 0.001
batch_size = 128

Train = True
# Train = False

patchsize = 25

hidden = int(sys.argv[1])
resize = int(sys.argv[2])
epochs = int(sys.argv[3])
csvfilepath = sys.argv[4]
# hidden = 20
# resize = 11


csvinputlen = patchsize * patchsize
networkinputlen = resize * resize
middle = int(csvinputlen / 2)

tau = 100000 # 300ms if use exp decay for preprocessing

prefix = 'aMSEO1H' + str(hidden) + '_linear' 

def preprocessingresize(features, resize,targetEventTS):
    

    features = features.transpose()
    # normalization
    featuresdiff = features - targetEventTS # features[middle]
    featuresNormed = (tau - np.abs(featuresdiff)) * 1.0 / tau
    featuresNormed = np.clip(featuresNormed, 0, 1)
    # featuresNormed = np.exp(-np.abs(featuresdiff)/tau)
    
    featuresNormed = featuresNormed.transpose()

    # crop
    features = featuresNormed.reshape(patchsize, patchsize)
    margin = int((patchsize - resize) / 2)
    cropend = patchsize - margin
    features = features[margin:cropend, margin:cropend]
    features = features.reshape(resize * resize)
    # print(featuresNormed.shape)
    return features

def preprocessing(features, targetEventTS):
    middle = int(patchsize * patchsize / 2)
    features = features.transpose()
    # normalization
    featuresdiff = features - targetEventTS
    featuresNormed = (tau - np.abs(featuresdiff)) * 1.0 / tau
    featuresNormed = np.clip(featuresNormed, 0, 1)
    # featuresNormed = np.exp(-np.abs(featuresdiff)/tau)

    featuresNormed = featuresNormed.transpose()
    return featuresNormed

train = {}
val = {}

class MyDataset():    

    def __init__(self, frame, transform=None):  

        self.data_frame = frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        m_data = self.data_frame.iloc[idx, :].values
        m_data = m_data.astype('float')

        if patchsize > resize:
            sample = {'y': m_data[1], 'x': preprocessingresize(m_data[2:2+csvinputlen], resize, m_data[0])}
        else:
            sample = {'y': m_data[1], 'x': preprocessing(m_data[2:2+csvinputlen], m_data[0])}

        if self.transform:
            sample = self.transform(sample)

        return sample





def splittraintest(csvdir, splitratio):
    allFiles = glob.glob(os.path.join(csvdir,'*TI25*.csv'))
    if len(allFiles) > 0:
        np_array_list = []
        for file_ in allFiles:
            print(file_)
            df = pd.read_csv(file_,usecols=[i for i in range(2,4+csvinputlen)], header=0)
            # if np.any(np.isnan(pd.DataFrame(df))):
            #     print('!!!!!!!!!!!!!!!!!!!!!!!there is nan')
            #     exit(0)
            # print(df.size)
            np_array_list.append(df.values)

        # read all csv files in a folder to one pandas frame
        comb_np_array = np.vstack(np_array_list)
        np.random.shuffle(comb_np_array)
        big_frame = pd.DataFrame(comb_np_array)
        # big_frame = big_frame.fillna(0)
        print(big_frame.head(), big_frame.size)
        # big_frame = big_frame.sample(frac=1)
        # print('shuffle done')

        # if np.any(np.isnan(big_frame)):
        #     print('!!!!!!!!!!!!!!!!!!!!!!!there is nan')
        #     exit(0)

        # msk = np.random.rand(len(big_frame)) > splitratio
        # print('mask done')
        # trainset = big_frame[msk]
        # testset = big_frame[~msk]

        # split train and test set
        leng = len(big_frame)
        trainset = big_frame.iloc[:int(splitratio*leng)]
        testset = big_frame.iloc[int(splitratio*leng):]
        del big_frame 
        # del msk
        return trainset, testset

traindata, testdata = splittraintest(csvdir=csvfilepath, splitratio=0.8)
trainset = MyDataset(traindata)
testset = MyDataset(testdata)

# datagenerator = DataGenerator(dataset, batch_size=batch_size, shuffle=True)
traingenerator = DataGenerator(trainset, batch_size=batch_size, shuffle=True)
testgenerator = DataGenerator(testset, batch_size=batch_size, shuffle=True)

print(len(traingenerator))
print(len(testgenerator))












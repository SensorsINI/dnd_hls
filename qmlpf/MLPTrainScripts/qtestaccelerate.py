from __future__ import print_function
from asyncio import current_task
from cmath import polar
from operator import mod
from sqlite3 import TimestampFromTicks
from time import time
# from keras import callbacks, models
import numpy as np
# from numpy.core.numeric import load
# np.random.seed(1337)
import os

from tensorflow.keras import optimizers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow
from tensorflow import keras
from tensorflow.keras import callbacks, models
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense,Embedding,Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import os,sys,glob
import pandas as pd

from sklearn.model_selection import train_test_split
from custom_dataloader.dataloader import DataGenerator

from qkeras import *
from qkeras.utils import model_save_quantized_weights, load_qmodel

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import scipy.io as sio

# training params
learning_rate = 0.0005
batch_size = 100#1024#128
patchsize = 25
csvinputlen = patchsize * patchsize
middle = int(csvinputlen / 2)


# hidden = int(sys.argv[1])
# resize = int(sys.argv[2])
# epochs = int(sys.argv[3])
# csvfilepath = sys.argv[4]

hidden = 10#20
resize = 7
epochs = 5
trainfilepath = 'D://qmlpfpys-s//2xTrainingDataDND21train//'
testfilepath = 'D://qmlpfpys-s//2xTrainingDataDND21test//'


# networkinputlen = resize * resize

tau = 128#128#100000 # 300ms if use exp decay for preprocessing

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
    plt.savefig(savename, format='png')
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
 
    def loss_plot(self, prefix, loss_type, e0loss, e0acc, e0valloss, e0valacc):

        iters = range(len(self.losses[loss_type])+1)
        np.save(prefix + 'loss.npy', np.array([e0loss] + self.losses[loss_type]))
        np.save(prefix + 'acc.npy', np.array([e0acc] + self.accuracy[loss_type]))

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
        
        plt.savefig(prefix+'training.pdf')
        plt.clf()


def preprocessingresize(allfeatures, resize, targetEventTS, targetEventP):
    # print(features.shape)
    # print(features)
    # print(allfeatures.shape, targetEventTS.shape, targetEventP.shape)
    absTS = allfeatures[:,:csvinputlen]
    polarity = allfeatures[:,csvinputlen:]
    
    features = absTS#.transpose()
    # print(features.shape, polarity.shape, targetEventTS.shape)
    # print(features[:10])
    # normalization
    # featuresdiff = pd.DataFrame(np.array(features)) - pd.DataFrame(np.array(targetEventTS)) 
    # print(featuresdiff.shape)
    featuresdiff = [features[i,:] - targetEventTS[i] for i in range(len(features))]
    # print(featuresdiff)
    featuresNormed = (tau - np.abs(featuresdiff)) * 1.0 / tau
    featuresNormed = np.clip(featuresNormed, 0, 1)
    # featuresNormed = np.exp(-np.abs(featuresdiff)/tau)
    
    # featuresNormed = featuresNormed.transpose()

    # crop
    features = featuresNormed.reshape(featuresNormed.shape[0], patchsize, patchsize)
    margin = int((patchsize - resize) / 2)
    cropend = patchsize - margin
    features = features[:,margin:cropend, margin:cropend]
    features = features.reshape(features.shape[0],resize * resize)

    polarity = polarity.reshape(polarity.shape[0],patchsize, patchsize)
    margin = int((patchsize - resize) / 2)
    cropend = patchsize - margin
    channelP = polarity[:,margin:cropend, margin:cropend]
    channelP = channelP.reshape(channelP.shape[0], resize * resize)
    channelP[features==0] = 0 # set the polarity to be 0 if the event is too old, which means the ts features are 0
    channelP[:,int(resize*resize/2)] = targetEventP # ensure the center location has the classified event's polarity
    

    features2 = np.hstack((features,channelP))
    # print(features2.shape)
    return features2



def preprocessingresizefortest(absTS,polarity, resize, targetEventTS, targetEventP):
    # print(features.shape)
    # print(features)
    # print(allfeatures.shape, targetEventTS.shape, targetEventP.shape)
    # absTS = allfeatures[:,:csvinputlen]
    # polarity = allfeatures[:,csvinputlen:]
    

    features = np.array(absTS)#.transpose()
    polarity = np.array(polarity)
    # print(features.shape, polarity.shape)
    # print(features.shape, polarity.shape, targetEventTS.shape)
    # print(features[:10])
    # normalization
    # featuresdiff = pd.DataFrame(np.array(features)) - pd.DataFrame(np.array(targetEventTS)) 
    # print(featuresdiff.shape)
    featuresdiff = np.abs([features[i,:] - targetEventTS[i] for i in range(len(features))])
    # print('featuresdiff', featuresdiff.shape)
    # print(featuresdiff)
    featuresdiff = (featuresdiff / 1000).astype(np.int)
    # featuresNormed = (tau - np.abs(featuresdiff)) * 1.0 / tau
    # featuresNormed = np.clip(featuresNormed, 0, 1)
    # featuresNormed[featuresNormed > 0] = 1

    featuresNormed = featuresdiff < tau
    featuresNormed = featuresNormed.astype(np.int)

    # crop
    features = featuresNormed.reshape(featuresNormed.shape[0], patchsize, patchsize)
    margin = int((patchsize - resize) / 2)
    cropend = patchsize - margin
    features = features[:,margin:cropend, margin:cropend]
    features = features.reshape(features.shape[0],resize * resize)

    polarity = polarity.reshape(polarity.shape[0],patchsize, patchsize)
    margin = int((patchsize - resize) / 2)
    cropend = patchsize - margin
    channelP = polarity[:,margin:cropend, margin:cropend]
    channelP = channelP.reshape(channelP.shape[0], resize * resize)
    channelP[features==0] = 0 # set the polarity to be 0 if the event is too old, which means the ts features are 0
    channelP[:,int(resize*resize/2)] = targetEventP # ensure the center location has the classified event's polarity
    

    features2 = np.hstack((features,channelP))
    # print(features2.shape)
    return features2


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

import math
def initializetsandpolmap(noiseRateHz, lastTimestampUs):
    # Random random = new Random()
    if True:
        for row in range(timestampImage.shape[0]):
            for col in range(timestampImage.shape[1]):
                p = random.random()
                t = -noiseRateHz * math.log(1 - p)
                tUs = (int) (1000000 * t)
                timestampImage[row][col] = lastTimestampUs - tUs
        
        
        for row in range(lastPolMap.shape[0]):
            for col in range(lastPolMap.shape[1]):
                b = random.random()
                # arrayRow[i] = b ? 1 : -1
                if b>0.5:
                    lastPolMap[row][col] = 1
                else:
                    lastPolMap[row][col] = -1

        lastPolMap[timestampImage < 0] = 0
        timestampImage[timestampImage < 0] = 0

            
        
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

        # if patchsize > resize:
        sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*2], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
        # else:
            # sample = {'y': m_data[4], 'x': preprocessing(m_data[5:5+csvinputlen*2], m_data[3])}

        if self.transform:
            sample = self.transform(sample)

        return sample

def BatchGenerator(files):
    for file_ in files:
        print('read file', file_)
        m_data = pd.read_csv(file_,usecols=[0] + [i for i in range(3,5+csvinputlen*2)], header=0).values

        print(m_data.shape)
        if patchsize > resize:
            # sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*2], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
            y = m_data[:,2]
            print(y.shape)
            x = preprocessingresize(m_data[:,3:3+csvinputlen*2], resize, m_data[:,1], m_data[:,0])
        else:
            # sample = {'y': m_data[:,4], 'x': preprocessing(m_data[:,5:5+csvinputlen*2], m_data[:,3])}
            y = m_data[:,4]
            x = preprocessing(m_data[:,5:5+csvinputlen*2], m_data[:,3])
        # if self.transform:
        #     sample = self.transform(m_data)
        

        yield(x,y)

import random as random
def getgeneratorbatches(files):
    # print('start generator')
    # while 1:
    if 1:
        # print('loop generator')
        sumbatches = 0
        # random.shuffle(files)
        # print(files)
        for file_ in files:
            try:
                # print(file_)
                # df = pd.read_csv(file_,usecols=[0] + [i for i in range(3,5+csvinputlen*2)], header=0)
                # df.fillna(0)
                df = pd.DataFrame(np.load(file_))
                zero = len(df[df.iloc[:,0] == 0])
                # random.shuffle(df.values)
                # print('read file', file_, zero, len(df)-zero)

                batches = int(np.ceil(len(df)/batch_size))
                sumbatches += batches
                
            except EOFError:
                print('error' + file_)
        print(sumbatches)
        return sumbatches   


sx = 260
sy = 346
patchsize = 7
timestampImage = np.zeros([sx,sy])
lastPolMap = np.zeros([sx,sy])

subsampleBy = 0
def getageandpolstringTI25(eventarray):
    if True:
        if True:
            if True:
                if True:
                    batchagechannel = np.zeros([eventarray.shape[0], patchsize * patchsize])
                    batchpolchannel = np.zeros([eventarray.shape[0], patchsize * patchsize])
                    # batchlabels = []
                    batchcount = 0
                    for event in eventarray:
                        # batchlabels.append(event[0]) # singalflag
                        ts = event[1] # event.timestamp
                        # type = event.getPolarity() == PolarityEvent.Polarity.Off ? -1 : 1
                        
                        x = (event[3] >> subsampleBy)
                        y = (event[2] >> subsampleBy)
                        type = event[4]
                        
                        radius = int((patchsize - 1) / 2)
                        if ((x < 0) or (x > sx) or (y < 0) or (y > sy)):
                            continue
                        
                        import time

                        absT= np.zeros([patchsize*patchsize,])#""
                        pols= np.zeros([patchsize*patchsize,])#""
                        indz = 0
                        stime1 = time.time()
                        xleftbound = max(0,x-radius)
                        xrightbound = min(sx,x+radius)
                        yleftbound = max(0,y-radius)
                        yrightbound = min(sy,y+radius)
                        # print(x,y,xleftbound,xrightbound,yleftbound,yrightbound)
                        croppedTI = timestampImage[xleftbound:xrightbound+1,yleftbound:yrightbound+1]
                        
                        # flattencroppedTI = np.reshape(croppedTI,order='F')
                        flattencroppedTI = croppedTI.flatten(order='F')
                        # print(timestampImage.shape, croppedTI.shape,flattencroppedTI.shape)

                        croppedpol = lastPolMap[xleftbound:xrightbound+1,yleftbound:yrightbound+1]
                        
                        # flattencroppedpol = np.reshape(croppedpol,order='F')
                        flattencroppedpol = croppedpol.flatten(order='F')
                        # print(lastPolMap.shape,croppedpol.shape,flattencroppedpol.shape)

                        startind = (yleftbound-(y-radius)) * radius + (xleftbound-(x-radius))
                        leng = flattencroppedpol.shape[0]
                        # print(flattencroppedpol.shape)
                        absT[startind:startind + leng,] = flattencroppedTI
                        pols[startind:startind + leng,] = flattencroppedpol
                        
                        etime1 = time.time()
                        
                        


                        # stime = time.time()
                        # realcount = 0
                        # for indy in range(-radius,radius+1):# (indx = -radius indx <= radius indx++) {
                        #     for indx in range(-radius,radius+1):#(indy = -radius indy <= radius indy++) {
                        #         absTs = 0
                        #         pol = 0
                        #         if ((x + indx >= 0) and (x + indx < sx) and (y + indy >= 0) and (y + indy < sy)):
                        #             absTs = timestampImage[x + indx][y + indy]
                        #             pol = lastPolMap[x + indx][y + indy]
                        #             realcount += 1
                                
                        #         # absT= absT+ absTs + "" + ","
                        #         # pol= pol+ pol + "" + ","

                        #         # absT.append(absTs)
                        #         # pols.append(pol)
                        #         absT[indz] = absTs
                        #         pols[indz] = pol
                        #         indz += 1
                        # etime = time.time()
                        # # print('old time',etime-stime)
                        # oldpols = pols
                        # print('old pols', pols)

                        # print('update number', realcount, 'new shape', startind,startind + leng,flattencroppedTI.shape)

                        
                        # print('new time',etime1-stime1)
                        # print('new pols', pols)
                        # print('equal', oldpols == pols)
                        


                        batchagechannel[batchcount] = absT
                        batchpolchannel[batchcount] = pols
                        batchcount += 1
                        timestampImage[x][y] = ts
                        lastPolMap[x][y] = type
                        

    return batchagechannel,batchpolchannel
                          
def mygeneratory(files):
    print('start generator')
    while 1:
        # print('loop generator')
        sumbatches = 0
        # random.shuffle(files)
        print(files)
        for file_ in files:
            try:
                # print(file_)
                # df = pd.read_csv(file_,usecols=[0] + [i for i in range(3,5+csvinputlen*2)], header=0)
                # df.fillna(0)
                df = pd.DataFrame(np.load(file_))
                df.fillna(0)
                
                zero = len(df[df.iloc[:,0] == 0])
                # random.shuffle(df.values)
                # print('read file', file_, zero, len(df)-zero)

                batches = int(np.ceil(len(df)/batch_size))
                sumbatches += batches
                for i in range(0, batches):
                    e_data = df[i*batch_size:min(len(df),i*batch_size+batch_size)].values
                    # print(m_data.shape,m_data[:10])
                    # e_data = e_data.astype('float')
                    # print(m_data.shape)
                    if patchsize >= resize:
                        # sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*2], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
                        y = e_data[:,0]
                        
                        # print(y.shape)
                        # agechannel,polchannel = getageandpolstringTI25(e_data)
                        # print(np.array(agechannel).shape)
                        
                        # print(np.array(polchannel).shape)
                        # print(agechannel)
                        # print(polchannel)
                        # print(e_data.shape, '**********************')
                        x = e_data[:,1]#preprocessingresizefortest(agechannel,polchannel, resize, e_data[:,1], e_data[:,4])
                        # print(e_data[:,1])
                        # print(e_data[:,4])
                        # print(x)
                        # exit(0)
                        # print(y.shape,y[:10])
                        # print(x.shape,x[:10])
                    else:
                        # sample = {'y': m_data[:,4], 'x': preprocessing(m_data[:,5:5+csvinputlen*2], m_data[:,3])}
                        y = e_data[:,0]
                        # x = preprocessing(m_data[:,5:5+csvinputlen*2], m_data[:,3])
                    # if self.transform:
                    #     sample = self.transform(m_data)
                    
                    # print(x[:10])
                    # print(y[:10])
                    # exit(0)
                    yield(x,y)
            except EOFError:
                print('error' + file_)
        print(sumbatches)   

                          
def mygenerator(files):
    print('start generator')
    while 1:
        # print('loop generator')
        sumbatches = 0
        # random.shuffle(files)
        print(files)
        for file_ in files:
            try:
                # print(file_)
                # df = pd.read_csv(file_,usecols=[0] + [i for i in range(3,5+csvinputlen*2)], header=0)
                # df.fillna(0)
                df = pd.DataFrame(np.load(file_))
                df.fillna(0)
                
                zero = len(df[df.iloc[:,0] == 0])
                # random.shuffle(df.values)
                # print('read file', file_, zero, len(df)-zero)

                batches = int(np.ceil(len(df)/batch_size))
                sumbatches += batches
                for i in range(0, batches):
                    e_data = df[i*batch_size:min(len(df),i*batch_size+batch_size)].values
                    # print(m_data.shape,m_data[:10])
                    # e_data = e_data.astype('float')
                    # print(m_data.shape)
                    if patchsize >= resize:
                        # sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*2], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
                        y = e_data[:,0]
                        
                        # print(y.shape)
                        agechannel,polchannel = getageandpolstringTI25(e_data)
                        # print(np.array(agechannel).shape)
                        
                        # print(np.array(polchannel).shape)
                        # print(agechannel)
                        # print(polchannel)
                        # print(e_data.shape, '**********************')
                        x = preprocessingresizefortest(agechannel,polchannel, resize, e_data[:,1], e_data[:,4])
                        # print(e_data[:,1])
                        # print(e_data[:,4])
                        # print(x)
                        # exit(0)
                        # print(y.shape,y[:10])
                        # print(x.shape,x[:10])
                    else:
                        # sample = {'y': m_data[:,4], 'x': preprocessing(m_data[:,5:5+csvinputlen*2], m_data[:,3])}
                        y = e_data[:,0]
                        # x = preprocessing(m_data[:,5:5+csvinputlen*2], m_data[:,3])
                    # if self.transform:
                    #     sample = self.transform(m_data)
                    
                    # print(x[:10])
                    # print(y[:10])
                    # exit(0)
                    yield(x,y)
            except EOFError:
                print('error' + file_)
        print(sumbatches)   

class GenDataset():    

    def __init__(self, data_path, transform=None):  

        # self.data_frame = frame
        allFiles = glob.glob(os.path.join(data_path,'*TI25*.csv'))
        # if 'TI25' in data_path:
        for file_ in allFiles:
            df = pd.read_csv(file_,usecols=[0] + [i for i in range(3,5+csvinputlen*2)], header=0) # might change if the collecting code change
            # np_array_list.append(df.values)
            m_data = df.values
            m_data = m_data.astype('float')
        # read all csv files in a folder to one wpndas frame
        # comb_np_array = np.vstack(self.data)
        # np.random.shuffle(comb_np_array)

        # m_data = self.data_frame.iloc[idx, :].values
        # m_data = m_data.astype('float')

        if patchsize > resize:
            sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*2], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
        else:
            sample = {'y': m_data[4], 'x': preprocessing(m_data[5:5+csvinputlen*2], m_data[3])}

        if self.transform:
            sample = self.transform(m_data)

        self.data = sample
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
        # return pd.DataFrame(np.vstack(np_array_list[:int(splitratio*len(np_array_list))])), pd.DataFrame(np.vstack(np_array_list[int(splitratio*len(np_array_list)):]))
        self.data_gen = self.get_data()
        self.transform = transform

    def get_data(self):
        for rec in self.data:
            batch_samples = []
            while len(batch_samples) > 0:
                yield batch_samples.pop()
        # return

    def __len__(self):
        return len(self.data * 4)

    def __getitem__(self, idx):
        # m_data = self.data_frame.iloc[idx, :].values
        # m_data = m_data.astype('float')

        # if patchsize > resize:
        # sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*2], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
        # else:
            # sample = {'y': m_data[4], 'x': preprocessing(m_data[5:5+csvinputlen*2], m_data[3])}

        # if self.transform:
            # sample = self.transform(sample)

        return next(self.data_gen)


def splittraintest(csvdir, splitratio):
    
    allFiles = glob.glob(os.path.join(csvdir,'*TI25*.csv'))
    if len(allFiles) > 0:
        np_array_list = []
        for file_ in allFiles:
            print(file_)
            df = pd.read_csv(file_,usecols=[0] + [i for i in range(3,5+csvinputlen*2)], header=0) # might change if the collecting code change
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

def buildModel(resize, hidden):
    networkinputlen = resize * resize
    if resize > 0:
        inputs = Input(shape=[networkinputlen*2, ], name='input')
        x = Dense(hidden, input_shape=(networkinputlen*2, ), activation='relu', name='fc1')(inputs)
        x = Dense(1, activation='sigmoid', name='output')(x)  
        model = Model(inputs, x)

        return model



def qbuildModel(resize, hidden,bits):
    networkinputlen = resize * resize
    if resize > 0:

        inputs = Input(shape=[networkinputlen*2, ], name='input')
        x = QActivation("quantized_bits("+str(bits)+")")(inputs)
        x = QDense(hidden, input_shape=(networkinputlen*2, ),
                kernel_quantizer=quantized_bits(bits),
                bias_quantizer=quantized_bits(bits),
           name="fc1")(x)
        x = QActivation("quantized_relu(bits="+str(bits)+")")(x)
        x = QDense(1,
                kernel_quantizer=quantized_bits(bits),
                bias_quantizer=quantized_bits(bits),
           name="fc2")(x)
        x = QActivation("quantized_tanh(16)",name='output')(x)

        model = Model(inputs, x)

        # inputs = Input(shape=[98, ], name='input')
        # x = QActivation("quantized_bits(3,0)")(inputs)
        # x = QDense(10, input_shape=(98,),

        #            kernel_quantizer=quantized_bits(3, 0),
        #            bias_quantizer=quantized_bits(3, 0),
        #            name="fc1")(x)
        # x = QActivation("quantized_relu(3,0)")(x)
        # x = QDense(1,

        #            kernel_quantizer=quantized_bits(3, 0),
        #            bias_quantizer=quantized_bits(3, 0),
        #            name="fc2")(x)

        # # x = smooth_sigmoid(x)
        # x = QActivation("quantized_tanh(16)", name='output')(x)

        # model = Model(inputs, x)

        return model

def qbuildModel2bit(resize, hidden):
    networkinputlen = resize * resize
    print('build 2 bits')
    bits = 2
    if resize > 0:
        inputs = Input(shape=[networkinputlen*2, ], name='input')
        x = QActivation("quantized_bits(2)", name="act0")(inputs)

        x = QDense(hidden, input_shape=(networkinputlen*2, ), kernel_quantizer=quantized_bits(bits),
           bias_quantizer=quantized_bits(bits),
           name="fc1")(x)  
        x = QActivation("quantized_relu(2)", name="relu1")(x)

        x = QDense(1, kernel_quantizer=quantized_bits(bits),
           bias_quantizer=quantized_bits(bits),
           name="fc2")(x)
        x = QActivation("quantized_bits(8)", name="qoutput")(x)
        x = Activation("sigmoid", name="output")(x)
        model = Model(inputs, x)

        return model


import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y_true,y_score,prefix):
    fpr,tpr,threshold = roc_curve(y_true,y_score,pos_label=1)
    auc = roc_auc_score(y_true,y_score)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('roc curve' + str(auc))
    plt.plot(fpr,tpr,color='b',linewidth=1)
    plt.plot([0,1],[0,1],'r--')
    plt.savefig(prefix + '_roccurve.pdf')
    plt.clf()



def getacc(y_true,initpredictions):
    y_pred = (initpredictions > 0.5).astype(int)
    y_true = np.reshape(y_true, [-1])
    y_pred = np.reshape(y_pred, [-1])
        

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def getmodel(mtype,resize,hidden,bits):
    if mtype == 'double':
        model = qbuildDModel(resize,hidden)
        middlefix = 'DH'
    elif mtype == 'qsingle':
        # if modelmode == '9bit':
        #     model = qbuildModel9bit(resize,hidden)
        # elif modelmode == '8bit':
        #     model = qbuildModel8bit(resize,hidden)
        # elif modelmode == '2bit':
        #     model = qbuildModel2bit(resize,hidden)
        # else:
        #     model = qbuildModel8bit(resize,hidden)
        model = qbuildModel(resize,hidden,bits)

        middlefix = 'qH'
    elif mtype == 'single':
        model = buildModel(resize,hidden)
        middlefix = 'fH'
    # elif mtype == 'perceptron':
    #     model = qbuildPerceptron(resize)
    #     middlefix = 'H'

    return model,middlefix

def trainFunction(traineventfile, testeventfile,  trainbatches, testbatches,resize, hidden, epochs, repeat, mtype, Train,wkfiles,bitsarr,flag):
    # global csvdir
    

    
    

    for bits in bitsarr:
        testgenerator = mygenerator(testfiles)
        testgeneratory = mygeneratory(testfiles)

        model,middle = getmodel(mtype,resize,hidden,bits)
        # testx,testy = getxandy(testfiles[0])
                
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        model.summary()

        # if True:
        for wkfile in wkfiles:
            print(wkfile)
            model.load_weights(wkfile)

            y_true = np.array([])
            initpredictions = np.array([])

            icount = 0
            # testbatches = 10
            print('predict batch',)
            y_true = np.load(flag+'ytrue.npy')[:testbatches*batch_size]
            initpredictions = model.predict_generator(testgenerator,steps=testbatches,verbose=1)
            # print(y_true.shape,initpredictions[:,0].shape)
            initpredictions= initpredictions[:,0]
            rocauc = roc_auc_score(y_true,initpredictions)
            print(wkfile.split('//')[-1],testfiles[0],'auc',rocauc)
            # plot_roc_curve(y_true,initpredictions,wkfile.replace('.h5',testfiles[0].replace('.npy', '') + str(rocauc)))
            plot_roc_curve(y_true,initpredictions,wkfile.replace('.h5','_'+flag+'_tbatch'+str(testbatches)+'_' + str("%.3f"%rocauc)))
            np.save(str(bits)+ 'bits_dri' + '_tbatch'+str(testbatches)+'_' + str("%.3f"%rocauc)+'.npy',np.concatenate([y_true,initpredictions],axis=0))
            
            continue

        
flag = sys.argv[1]



if flag == 'dri':
    trainfiles = ['driandrealshotnoiseevents.npy']
    testfiles = ['driandrealshotnoiseevents.npy']
    trainbatches = 62000
    testbatches = 62000

if flag == 'hotel':
    trainfiles = ['hotelandrealshotnoiseevents1.npy']
    testfiles = ['hotelandrealshotnoiseevents1.npy']
    trainbatches = 31000
    testbatches = 31000


testbatches = int(sys.argv[2])
lastts = np.load(testfiles[0])[0,1]
print('lastts', lastts)


initializetsandpolmap(5,lastts)



wkfiles1 = []
# wkfiles1 = ['0314l216bitsqtanhputbin100tau' + str(bits) + 'bitaw' + str(repeat) + 'MSEO1' + middle + '10_linear_7weights.h5']
wkfiles1 = ['software-0314-tanhmodel.h5']
# bits = [i for i in range(2,9)]
bits = [3]
middle = 'qsingle'
for repeat in [0]:
    trainFunction(trainfiles[0],testfiles[0],  trainbatches,testbatches, resize, hidden, epochs, repeat, middle, False,wkfiles1,bits,flag)


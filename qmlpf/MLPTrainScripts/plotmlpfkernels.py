import numpy as np
# from numpy.core.numeric import load
np.random.seed(42) #42 best 6 worst
import os

import matplotlib.pyplot as plt
import numpy as np


from tensorflow.keras import optimizers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow
from tensorflow import keras
from tensorflow.keras import callbacks, models
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense,Embedding,Dropout,Conv2D,Flatten,MaxPooling2D
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

# from weighted_binary_cross_entropy import weighted_binary_crossentropy
# from gabor_initializer_1d_patch import gabor_initializer_1d_patch

from easygui import fileopenbox, integerbox, diropenbox

# https://stackoverflow.com/questions/68992738/can-we-get-weights-by-name-in-keras
def get_weights_by_name(model, name):
    if hasattr(model,'weights') and not model.weights is None:
        return [w for w in model.weights if w.name==name][0]
    if hasattr(model,'variables') and  not model.variables is None:
        return [w for w in model.variables.weights if w.name==name][0]

def plot_kernels(model=None, dir=None, image_file_name='kernels.png', normalize_each=True):
    ''' Plots hidden layer kernels for MLP
    
    :param model: plots these weights unless it is None
    :param dir: loads TF2 model from this dir if model is None
    :param image_file_name: puts plot as PNG in dir/image_file_name unless dir is None, then just puts to image_file_name
    :param normalize_each: True to normalize each kernel individually, False to normalize wrt all weights
    
    '''

    img_width = 7
    img_height = 7

    # custom_objects={"_weighted_binary_crossentropy": weighted_binary_crossentropy()}
    # custom_objects={"_weighted_binary_crossentropy": weighted_binary_crossentropy(), "_gabor_initializer_1d_patch": gabor_initializer_1d_patch()}

    if model is None:
        # model = tensorflow.keras.models.load_model(modelpath, custom_objects=custom_objects)
        model = tensorflow.keras.models.load_model(dir)
        # model.summary()
        layer_name = 'fc1'
        # layer = model.get_layer(name=layer_name)
        # weights =layer.get_weights()[0]
        # print(weights.shape)
    # weights=model.variables.weights[0]
    # weights=model.layers[0].weights[0]
    weights=get_weights_by_name(model,'fc1/kernel:0')
    output_weights=get_weights_by_name(model,'output/kernel:0')

    # model = keras.models.load_model(modelpath)
    # model.summary()
    # layer_name = 'fc1'
    # layer = model.get_layer(name=layer_name)
    # weights =layer.get_weights()[0]
    # print(weights.shape)
    # print(weights)

    num_kernel_pairs = weights.shape[1]
    # print(num)
    # weights = np.reshape(weights,(num_kernel_pairs,2,7,7))
    # weights have shape e.g. (98,20), the first index is pixels for age, pixels for pol, the 2nd index is kernel number (in this case 20, i.e. 20 age and 20 polarity)
    weights = np.reshape(weights,(7,7,num_kernel_pairs*2), order='F') # note order F means to read / write the elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest. 

    print(f'hidden layer weights shape: {weights.shape}')
    print(output_weights)

    num_col=int(np.ceil(np.sqrt(2*num_kernel_pairs)))+1
    fig = plt.figure(figsize=(12, 9))
    if not dir is None: 
        ftitle=os.path.basename(dir)
    else:
       ftitle=model.name
    
    ftitle=ftitle+(' indiv. norm' if normalize_each else ' global norm')
    
    plt.suptitle(ftitle)
    fontsize=7
    flags = ['age','pol']
    weight_min=np.min(weights[:])
    weight_max=np.max(weights[:])
    for i in range(1, num_kernel_pairs*2+1):
        ax = fig.add_subplot(num_col, num_col, i)
        weight = weights[:,:,i-1]
        if normalize_each:
            weight = (weight-np.min(weight))/(np.max(weight)-np.min(weight))
        else:
            weight = (weight-weight_min)/(weight_max-weight_min)
            
        ax.imshow(weight, cmap='gray')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        flag = flags[int(i%2)]
        kernel_num = int((i-1)/2)
        tname = f'{kernel_num}: {flag} {float(output_weights[kernel_num]):.2f}' 
        print(tname)
        ax.set_title(tname, fontsize=fontsize)
    
    fname=os.path.join(dir,image_file_name) if (not dir is None) else image_file_name
    plt.savefig(fname)
    print('saved '+fname)
    plt.show()
        
if __name__ == "__main__":
    from prefs import MyPreferences
    prefs=MyPreferences()


    # f=fileopenbox('select h5 model file (not the dated final weights h5 file)', default=prefs.get("lastfile",'models/*.h5'),title='h5 chooser')
    dir=diropenbox('select model dir', default=prefs.get("lastfile",'models/'),title='model dir chooser')
    if dir is None or dir=='':
        print('no file selected')
        quit(0)
    prefs.put('lastfile',dir)
    plot_kernels(None, dir,'kernels.png')


## This file is part of https://github.com/SensorsINI/dnd_hls.
## This intellectual property is licensed under the terms of the project license available at the root of the project.

# use tf2.x(tf1.14 also ok) to convert the model hdf5 trained by tf2.x to pb
import tensorflow as tf
import os,sys


# import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense,Embedding,Dropout,Activation
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
# def create_model():

#     base_model=ResNet50(include_top=True, weights=None, classes=2)

#     model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

#     return model

def create_model(resize, hidden):
    networkinputlen = resize * resize
    if resize > 0:
        inputs = Input(shape=[networkinputlen*2, ], name='input')
        x = Dense(hidden, input_shape=(networkinputlen*2, ), activation='relu', name='fc1')(inputs)
        # x = QDense(hidden, input_shape=(networkinputlen*2, ), kernel_quantizer=quantized_bits(8,0,1),
        #    bias_quantizer=quantized_bits(8,0,1),
        #    name="fc1")(inputs)
        # x = QActivation("quantized_relu(8,0)", name="relu1")(x)
        # x = Dropout(0.2)(x)
        x = Dense(1, name='fc2')(x)
        # x = QDense(1, kernel_quantizer=quantized_bits(8,0,1),
        #    bias_quantizer=quantized_bits(8,0,1),
        #    name="fc2")(x)
        x = Activation("sigmoid", name="output")(x)
        model = Model(inputs, x)

        return model


model=create_model(7,20)

# 编译模型

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


weightsh5filename = sys.argv[1]
model.load_weights(weightsh5filename)

modelh5filename = weightsh5filename.replace('.h5', 'model.h5')
model.save(modelh5filename)

def freeze_session(model_path=None,clear_devices=True):
    tf.compat.v1.reset_default_graph()
    session=tf.compat.v1.keras.backend.get_session()
    graph = session.graph
    with graph.as_default():
        model = tf.keras.models.load_model(model_path)
        output_names = [out.op.name for out in model.outputs]
        print("output_names",output_names)
        input_names =[innode.op.name for innode in model.inputs]
        print("input_names",input_names)
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            print('node:', node.name)
        print("len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)#去掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("length of  node",len(outgraph.node))
        (filepath,filename) = os.path.split(model_path)
        tf.io.write_graph(frozen_graph, "./2xpb/", filename.replace('.h5', '.pb'), as_text=False)
        return outgraph

def main(h5_path):  
    if not os.path.isdir('./2xpb/'):
        os.mkdir('./2xpb/')
    freeze_session(h5_path,True)

# main(sys.argv[1])
main(modelh5filename)

import tensorflow
from tensorflow import keras
from keras import layers, models
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Dropout, Flatten,Layer

from weighted_binary_cross_entropy import weighted_binary_crossentropy
from gabor_initializer_1d_patch import gabor_initializer_1d_patch


@keras.saving.register_keras_serializable(package="MyMLP")
class MyMLP(Sequential):

  def __init__(self, patch_size=7, num_hidden_units=20, num_hidden_layers=1, dropout=0, **kwargs):
    super(MyMLP,self).__init__(**kwargs)
    self.gabor_init = gabor_initializer_1d_patch(
        patch_height=patch_size,
        patch_width=patch_size,
        num_filters=num_hidden_units
    )
    self.custom_objects={"_weighted_binary_crossentropy": weighted_binary_crossentropy(), "_gabor_initializer_1d_patch": gabor_initializer_1d_patch()}
    self.inp_shape=(patch_size*patch_size*2, )
    self.add(Input(shape=self.inp_shape, dtype="float32", name="input"))
    # self.add(Flatten())
    if dropout>0:
        self.add(Dropout(dropout))
    # self.add(Flatten())
    self.add(Dense(units=num_hidden_units, 
              activation='relu', name='fc1', 
              kernel_initializer=self.gabor_init))
    if dropout>0:
        self.add(Dropout(dropout))
    self.add(Dense(1, activation='sigmoid', name='output'))

  def get_config(self):
    # config=super.get_config() # Sequential does not have get_config()
    # config.update(self.custom_objects)
    # return config
    return self.custom_objects
    
  # @classmethod
  # def from_config(cls, config):
  #     return cls(**config)
  
if __name__ == "__main__":
    model=MyMLP()
    from plotmlpfkernels import plot_kernels
    plot_kernels(model,"mymlp-export")
    print(str(model))
    
    # model.build(input_shape=model.inp_shape)
    model.save("mymlp-save")
    model.save("mymlp-save.h5",save_format='h5')
    model.save("mymlp-save.keras",save_format='keras')
    model.export("mymlp-export")
    
    # test saving
    custom_objects={"_weighted_binary_crossentropy": weighted_binary_crossentropy(), "_gabor_initializer_1d_patch": gabor_initializer_1d_patch()}
    loaded_model=tensorflow.keras.models.load_model("mymlp-export")
    

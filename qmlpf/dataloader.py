import numpy as np
from keras.models import Sequential
import DataGenerator
# Parameters
params = {'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
# Generators



training_generator = DataGenerator(train_df, train_idx, **params)
validation_generator = DataGenerator(val_df, val_idx, **params)
 
 
# Design model
model = Sequential()
[...] # Architecture
model.compile()
 
 
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=4)


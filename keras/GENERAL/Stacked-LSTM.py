
# coding: utf-8
#import comet_ml in the top of your file
from comet_ml import Experiment

#create an experiment with your api key
experiment = Experiment(api_key="Jrmp1SbY5izsv7D1PWMMRpDGD",
  log_code=True,
  project_name='keras-demos',
  auto_param_logging=True)

'''
Stacked LSTM for sequence classification

In this model, we stack 3 LSTM layers on top of each other,
making the model capable of learning higher-level temporal representations.

The first two LSTMs return their full output sequences,
but the last one only returns the last step in its output sequence,
thus dropping the temporal dimension
(i.e. converting the input sequence into a single vector).
'''

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
epochs = 5

params={"data_dim":data_dim,
"timesteps":timesteps,
"num_classes":num_classes,
"epochs":epochs,
"layer1":"LSTM(64)",
"layer2":"LSTM(64)",
"layer3":"LSTM(64)",
}

experiment.log_multiple_params(params)
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(64, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))


model.fit(x_train, y_train,
          batch_size=64, epochs=epochs,
          validation_data=(x_val, y_val))


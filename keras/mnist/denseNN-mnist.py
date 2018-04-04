# coding: utf-8
#import comet_ml in the top of your file
from comet_ml import Experiment

#create an experiment with your api key
experiment = Experiment(api_key="Jrmp1SbY5izsv7D1PWMMRpDGD",project_name='mnist')

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

batch_size = 128
num_classes = 10
epochs = 1


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

params={"batch_size":batch_size,
"epochs":epochs,
"layer1":"Dense(64)",
"layer1-activation":"relu",
"optimizer":"adam",
"auto_param_logging":True
}

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
#model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

print(model.summary()) #always a good idea to print/preserve the model summary

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

with experiment.train():
  history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=3, verbose=1, mode='auto')])

with experiment.test():
  loss, accuracy = model.evaluate(x_test, y_test)
  print(loss, accuracy)
  metrics = {
    'loss':loss,
    'accuracy':accuracy
  }
  experiment.log_multiple_metrics(metrics)



experiment.log_multiple_params(params)
experiment.log_dataset_hash(x_train) #creates and logs a hash of your data

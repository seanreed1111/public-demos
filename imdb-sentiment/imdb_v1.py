# -*- coding: utf-8 -*-
import numpy as np
#from keras.datasets import imdb
import pickle

skip_top=30
maxlen=300
random_seed = 42

try:
  with open("imdb_data.pickle",'rb') as f:
    data = pickle.load(f)

except FileNotFoundError:
  print("Error Loading from file. Loading from keras now...")
  (X_train, y_train), (X_test, y_test) = imdb.load_data(
    skip_top=skip_top,
    maxlen=maxlen,
    seed=random_seed)

  data = {
    "X_train":X_train,
    "X_test":X_test,
    "y_train":y_train,
    "y_test":y_test
  }
  with open("imdb_data.pickle",'wb') as f:
    pickle.dump(data, f)

print("data['X_train'].shape = ", data["X_train"].shape)
print("data['X_test'].shape = ", data["X_test"].shape)

print("number of output classes in training set", np.unique(data["y_train"]))
print("Bincount of training set ", np.bincount(data["y_train"]))
print("\n\nnumber of output classes in test set", np.unique(data["y_test"]))
print("Bincount of test set ", np.bincount(data["y_test"]))

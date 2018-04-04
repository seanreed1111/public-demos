
# coding: utf-8

# **Chapter 3 – Classification**
#
# _This notebook contains all the sample code and solutions to the exercices in chapter 3 of "Hands On ML With Scikit Learn and TensorFlow"._

# # Setup

# First, let's make sure this notebook works well in both python 2 and 3, import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures:

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
#import comet_ml in the top of your file
from comet_ml import Experiment

#create an experiment with your api key
experiment = Experiment(api_key="Jrmp1SbY5izsv7D1PWMMRpDGD", log_code=True, project_name='sklearn-demos')

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

# # MNIST

# In[2]:


from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape, digits.target.shape)


print(digits.DESCR)

X, y = digits.data, digits.target
print(X.shape, y.shape)

# # Binary classifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                   random_state=42)

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


# In[16]:


log_reg = LogisticRegression(random_state=42)


# In[17]:


log_reg.fit(X_train, y_train_5)


# In[18]:


y_pred = log_reg.predict(X_test)


# In[19]:


target_names=["not 5", "5"]
print(classification_report(y_test_5, y_pred, target_names=target_names))





# In[53]:



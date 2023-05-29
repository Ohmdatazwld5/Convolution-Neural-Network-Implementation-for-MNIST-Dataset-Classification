#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Input
from keras.layers import Conv2D, MaxPool2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler


# In[4]:


(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()


# In[5]:


x_train=x_train/255.0
y_train=tf.keras.utils.to_categorical(y_train)


# In[6]:


t=(x_train[0])
plt.imshow(t)


# In[7]:


model= Sequential()


# In[8]:


model.add(Input(shape=(28,28,1),name='input_layer'))
model.add(Conv2D(filters=16,kernel_size=(2,2),strides=(1,1),activation=tf.keras.activations.relu,name='Conv2D_Layer1a'))
model.add(Conv2D(filters=16,kernel_size=(2,2),strides=(1,1),activation=tf.keras.activations.relu,name='Conv2D_Layer1b'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='Pooling_Layer'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),activation=tf.keras.activations.relu,name='Conv2D_Layer2a'))
model.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),activation=tf.keras.activations.relu,name='Conv2D_Layer2b'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='Pooling_Layer1'))
model.add(Flatten(name='Flattening_Layer'))
model.add(Dense(16,activation=tf.keras.activations.relu,name='Hidden_Layer1'))
model.add(Dense(10,activation=tf.keras.activations.softmax))
model.summary()


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics='acc')


# In[24]:


hist=model.fit(x_train,y_train,batch_size=64,validation_split=0.1,epochs=12)


# In[25]:


model.predict(x_test)


# In[26]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[27]:


x_test=x_test/255.0
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)


# In[28]:


accuracy_score(y_test,y_pred)


# In[29]:


confusion_matrix(y_test,y_pred)


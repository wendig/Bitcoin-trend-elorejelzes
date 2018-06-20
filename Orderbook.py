# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:33:04 2018
    
    Data generator, yield example
    https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
    datageneration, elore legeneral az osszeset
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    
    fit generation
    https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator    
    DATA SHAPE
TODO:
    generator helyett lehet, kepek elkeszitese
    
@author: lorand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import random

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Conv2D
from keras.layers import Flatten, MaxPooling2D
from keras import optimizers
from keras.optimizers import SGD


###############################################################################
# Functions

def data_generator(X,y, size=20):
    '''
    X: orderbook matrix
    y: label
    size: number of orderbook we pass
    '''
    x_len = X.shape[0] - size - 1 
    # _index = 0
    _index = random.randint(0, x_len)
    while True:
        i = _index % x_len
        
        _index += 1
        yield X[i:i+size,:],y[i + size]


def batch_generator(X, y, batch_size = 50 ,size=20):
    '''
    tutorial: https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
    X: orderbook matrix
    y: label
    size: number of orderbook are in a picture
    batch_size: number of picture we send
    '''
    # (nb_sample, height, width, channel)
    batch_features = np.zeros((batch_size, size,X.shape[1], 1))
    batch_labels = np.zeros((batch_size,1))
    
    # _index = 0
    kulso = random.randint(0, X.shape[0] - (size+batch_size) - 2 )
    while True:
        for j in range(batch_size):                
            # get a picture
            data = X[kulso+j:kulso+size+j,:]
            # reshape and append to batch_features
            batch_features[j] = data.reshape((data.shape[0],data.shape[1],1))
            # append label
            batch_labels[j] = y[kulso+size+j]
            
        kulso += batch_size
        
        vegso_index = kulso + size + batch_size + 2
        if  vegso_index > X.shape[0]:  
            kulso = vegso_index % X.shape[0]#ide uj rand generator is lehet
            
        yield batch_features, batch_labels
        
        
def window(x, n):
    #https://stackoverflow.com/questions/10849054/array-of-neighbor-averages
    return (x[(n-1):] + x[:-(n-1)])/float(n)



def cut(in_matrix,_from, _to):
    return in_matrix[_from:_to,:]


def create_pictures(data: np.ndarray, label: np.array, height: int):
    
    features = np.zeros((data.shape[0]-height, height, data.shape[1], 1))
    
    len_data = data.shape[0]
    for i in range(0,len_data-height):
        picture = data[i:i+height,:]
        
        # TODO !!! MASHOVA ATRAKKKKK PLSSSSS
        
        
        
        features[i] = picture.reshape((picture.shape[0],picture.shape[1],1))# CHECK
        
    return features,label[height:]

def split_data(data, training_size=0.8):
    return data[:int(training_size*len(data))], data[int(training_size*len(data)):]


###############################################################################
# Preprocessing

df = pd.read_csv('orderbooks_3.csv')

# TODO
# reduce orderbook width
#i_width = 640
#i_height = 480
#scipy.misc.imresize(original_image, (i_height, i_width))
print('orderbook shape: {}'.format(df.values.shape))


######x
# Constants
b_size = 50 # batch_size
o_size = 40 # hight
width = df.values.shape[1]
epochs = 50


#df_price = pd.read_csv('prices_3.csv',nrows=400)
df_price = pd.DataFrame({'x':np.arange(df.values.shape[0])})
y = df_price.values
y = y / y.max()
###

features, labels = create_pictures(df.values,y,40)

x_train, x_test = split_data(features)
y_train, y_test = split_data(labels)

print(features.shape)
print(labels.shape)


###############################################################################
# Model
input_shape = ( o_size, width, 1) # channels last


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape= input_shape   ))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=b_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

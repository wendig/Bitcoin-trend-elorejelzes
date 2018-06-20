# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:57:25 2018
Price_lstm:
    A kovi x atlaga nagyobb mint az elozo x Atlaga
    1 - igen
    0 - hiban belul ugyan akkora
    -1 - nem
    ADATB formalas: 
        1. uj kategoria, kovi atlagok nagyobbak v kisebbek-e mint elozo
Orderbook:
    szinten 1,0,-1 kategoriakat josol
    y price, x 10 db orderbooksi, ezen 2d conv
    
https://medium.com/@siavash_37715/how-to-predict-bitcoin-and-ethereum-price-with-rnn-lstm-in-keras-a6d8ee8a5109
https://github.com/sudharsan13296/Bitcoin-price-Prediction-using-LSTM/blob/master/Bitcoin%20price%20prediction%20(Time%20series)%20using%20LSTM%20RNN.ipynb
@author: lorand
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras.optimizers import SGD
###############################################################################
# FUNCTIONS
dropout = 0.3
loss = 'mean_squared_error'
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
activ_func = 'tanh'


def build_model(inputs, output_size, neurons, dropout=dropout,\
                activation=activ_func, loss=loss, optimizer=sgd):
  model = Sequential()
  model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
  model.add(Dropout(dropout))
  model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
  model.add(Dropout(dropout))
  model.add(LSTM(neurons, activation=activ_func))
  model.add(Dropout(dropout))
  model.add(Dense(units=output_size))
  #model.add(Activation(activ_func))
  model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
  model.summary()
  return model
"""
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
"""

def split_data(data, training_size=0.8):
    return data[:int(training_size*len(data))], data[int(training_size*len(data)):]

def gen_y(df,hossz):
    # y eloallitasa ide
    return 1
    
def create_dset(dataset, look_back=1):
    # dataset is a numpy matrix
    dataX, dataY = [], []
    for i in range(dataset.shape[0]-look_back-1):
        # dataset cols: x ema trend y
        a = dataset[i:(i+look_back),:3]
        b = a.flatten()
        dataX.append(b)#!!
        dataY.append(dataset[i + look_back - 1,3])
    return np.array(dataX), np.array(dataY)

###############################################################################


df = pd.read_csv('prices_3.csv',nrows=400)
#df = pd.DataFrame({'x':np.arange(500)})

y = df.values[:,0]
x = np.arange(len(y))
print(len(y))
plt.plot(x,y)
plt.show()


### New features
hossz = 50


# exponential moving average 
df['ema'] = df['x'].ewm(span=hossz).mean() 

# trend
df['tmp'] = df['ema'].diff()
df['trend'] = ( df['tmp'] - df['tmp'].mean() )/df['tmp'].std()# normalizal
df['trend'] = df['trend'] / df['trend'].abs().max() # 1 es -1 koze


y = df['trend'].values
x = np.arange(len(y))
plt.plot(x,y)
plt.title('Trend')
plt.show()

# drop helper column and 
df = df.dropna(axis=0, how='any')


# calc y
df['sma'] = df['x'].rolling(window=hossz).mean() # simple moving average 
# mozgo atlag a jovobeli adatokra:
df['x_rev'] = df['x'].values[::-1]
df['rev_sma'] = df['x_rev'].rolling(window=hossz).mean().values[::-1] 
# multbeli es jovobeli mozgo atlag kulonbsege, h novo v csokken a trend
df['y'] = df['rev_sma'] - df['sma']
# drop helper column and Nan rows
df = df.drop(['tmp','sma','x_rev','rev_sma'], axis=1)# !!!!
df = df.dropna(axis=0, how='any')


# Normalize
for col in df.columns:
    if col != 'y' and col != 'trend':
        df[col] = df[col] / df[col].max()
    if col == 'y':
        #df['y'] = ( df['y'] - df['y'].mean() )/df['y'].std()# normalizal
        #df['y'] = df['y'] / df['y'].abs().max() # 1 es -1 koze
        df[col] = df[col] / df[col].abs().max() # TODO !! ez a legjobb?
        
print(df.head(5))

y = df['y'].values
x = np.arange(len(y))
plt.plot(x,y)
plt.show()

df.to_csv('test.csv',index=False)


# Convert problem to supervised
# X: x,ema,trend, Y: y 



train, test = split_data(df.values)
x_train, y_train = create_dset(train,20)
x_test, y_test = create_dset(test,20)


tmp = pd.DataFrame(x_train)
tmp.to_csv('tets_xtrain.csv')

#[samples, timesteps, features].
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

print('x train shape = {}'.format(x_train.shape))
print('x test shape = {}'.format(x_test.shape))

print('y train shape = {}'.format(y_train.shape))
print('y test shape = {}'.format(y_test.shape))


print('start training')
# design network
model = Sequential()
model.add(LSTM(300, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer=sgd)# 'adam'
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model = build_model(x_train,1,50)

# fit network
history = model.fit(x_train, y_train, epochs=100, batch_size=72,\
                    validation_data=(x_test, y_test),\
                    verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

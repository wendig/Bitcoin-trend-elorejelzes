# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:33:00 2018

# orankenti keresek szama : 1225

@author: lorand
"""

import pandas as pd
import requests
from time import sleep
import csv
import json
import numpy as np
import scipy.misc
import copy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import traceback
import matplotlib.pyplot as plt
import time
from keras.optimizers import SGD


###############################################################################
# Constats
len_max_list = 8000 # a bids es asks is max ennyi ajanlatot tartalmazhat
max_orderbook = 200000 #150000 # Maximum number of orderbooks this script downloads
sleep_time = 1 # 2 !!!!  -> feldolgozasi ido 1.3 sec kb
ex_time = sleep_time/10 # If error occours from network, wait this time
maximum = 0
minimum = 1000000
size = 0


base_addr = "https://api.cryptowat.ch/"

prices = [] # stores downloaded prices
orderbooks = [] # stores downloaded processed orderbooks

name = '_{}'.format(sleep_time) # save files with this name
###############################################################################

###############################################################################
# Functions

def kerekit(lista):
    for elem in lista:
        elem[0] = int(elem[0])
        
def duplazott_torol(lista):
    for i in range(0,len(lista) - 1 ):
        if lista[i][0] == lista[i+1][0]:
            lista[i+1][1] += lista[i][1]
            lista[i][1] = 0

def torol_levag(lista):
    tmp = []
    for elem in lista:
        if elem[1] != 0:
            tmp.append(elem)
    return tmp

def pre_process(in_ask,in_bid):
    # ask
    kerekit(in_ask)
    duplazott_torol(in_ask)
    # bid
    kerekit(in_bid)
    duplazott_torol(in_bid)
    return torol_levag(in_ask), torol_levag(in_bid)

def summaz(lista):
    for j in range(1,len(lista)):
        lista[j][1] += lista[j-1][1]




def get_json(address):
    r = requests.get(address)    
    return r.json()

def get_max(ask):
    return ask[-1][0]

def get_min(bid):
    return bid[-1][0]

def set_min_max_size():
    """
    megkeresi az orderbook max es min ertekeit, skalazas a tovabbiakban ez alapjan fog tortenni
    TODO korlatok+!!!!
    """
    j = get_json()
    a = get_max(j["result"]["asks"])
    b = get_min(j["result"]["bids"])    
    return a, b, (a - b) + 1


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def bennevan(x):
    if x >= minimum and x <= maximum:
        return 1
    else:
        return 0


def osszefuz(ask,bid):
    tmp = zerolistmaker(size)
    
    
    #ASk
    for elem in ask:
        index = elem[0] - minimum
        
        if bennevan(elem[0]) == 1:
            try:
                tmp[index] += elem[1]
            except:
                #for debug
                traceback.print_exc()
                print(index)
                print(len(tmp))
                print('-')
                print("minimum = {}, maximum = {}".format(minimum,maximum))
                print("--------------")
    
    #BIDS
    for elem in bid:
        index = elem[0] - minimum
        
        if bennevan(elem[0]) == 1:
            tmp[index] -= elem[1]

    
    # eddig csak csucsok -> kisimit
    bid_max_index = bid[0][0] - minimum
    bid_min_index = 0
    for j in reversed(range(bid_min_index, bid_max_index )):
        if tmp[j] == 0:
            tmp[j] = tmp[j+1]
            
    ask_max_index = ask[-1][0] - minimum
    ask_min_index = ask[0][0] - minimum
    
    
    for j in range(ask_min_index, min(ask_max_index,len(tmp)) ):
        if tmp[j] == 0:
            tmp[j] = tmp[j-1]

    return tmp

def levag_egy(adatok,limit):
    ret = []    
    for orderbook in adatok:
        tmp = []
        for elem in orderbook:
            if elem[1] < limit: #  !!
                tmp.append(elem)
        ret.append(tmp)
    return ret


def set_new_len(lista,newlen):
    regi_len = len(lista)
    if newlen >= regi_len:
        print('regilen = {}, newlen = {}'.format(regi_len,newlen))
        print('-------')
        print(lista)
        print('-------')
        return 0
    for i in range(0,regi_len - newlen):
        del lista[-1]
    return 1
    


def polling_list(in_list,value):
    ret = []
    osszeg = 0
    
    for i in range(1,len(in_list)+1):
        if (i) % value == 0:
            ret.append(osszeg + in_list[i-1])
            osszeg = 0
        else:
            osszeg += in_list[i-1]
            
    
    return ret

def reduce_orderbook(obook,val):
    new_obook = []
    
    for i in range(len(obook)):
        new_obook.append( polling_list(obook[i],val) )
    return new_obook


def prep_data(lista_matrix):    
    adat = np.array(lista_matrix)
    return adat.T[0], adat.T[1]

def plot(lista_matrix,label,flip = False):
    x, y = prep_data(lista_matrix)
    if flip:    
        
        plt.plot(x,y)
    else:
        plt.plot(x,y)
    plt.title(label)
    plt.show()
    
###############################################################################

#################################
# START OF PROGRAM
# |
# v
    
    
##################
# Downloading Loop


# set size & min & max

j = get_json(base_addr + "markets/gdax/btcusd/orderbook")
print("cost of first request: {}".format(j["allowance"]["cost"]))


_ask = j["result"]["asks"]
_bid = j["result"]["bids"]
print("ask_len = {}, bid_len {}".format(len(_ask),len(_bid)))

set_new_len(_ask,len_max_list)
set_new_len(_bid,len_max_list)
ask, bid = pre_process(_ask,_bid)

summaz(ask)
summaz(bid)

plot(bid,'bid')
plot(ask,'ask')

# Set max, min, size
minimum = get_min(bid)
maximum = get_max(ask)
size = maximum - minimum + 1
print("minimum = {}, maximum = {}".format(minimum,maximum))

asd = osszefuz(ask,bid)


if maximum == 0 and minimum == 1000000:
    print("Minimum, maximum beallitasa hibas")


print("Start Downloading...")

error = 0

for i in range(0,max_orderbook):
    try:
        #price
        j_price = get_json(base_addr + "markets/gdax/btcusd/price")
        if "error" in j_price:
            print("out of allowance")
            break
        
        
        
        
        #orderbooks
        j = get_json(base_addr + "markets/gdax/btcusd/orderbook")
        if "error" in j:
            print("out of allowance")
            break
        
        
        _ask = j["result"]["asks"]
        _bid = j["result"]["bids"]
        
        # Data processing
         
        a = set_new_len(_ask,len_max_list)
        b = set_new_len(_bid,len_max_list)
        if a == 0 or b == 0:
            raise Exception('wtf length(assume download is not correct)')
            break
        
        ask, bid = pre_process(_ask,_bid)
        summaz(ask)
        summaz(bid)
        
        # Ask and Bid into one list
        asd = osszefuz(ask,bid)
        orderbooks.append(asd)


        # APPEND PRICE IF ORDERBOOK IS NICEE
        prices.append(j_price["result"]["price"])# todo Ar helyett, summary letoltese
        #print and sleep    
        if i % 50 == 0:
            print('request number {}, remaining {}'.format(i,j['allowance']['remaining']))
            
        
        error = 0
        sleep(sleep_time)
    except Exception as e:
        print(e)
        traceback.print_exc()
        if error >= 500:
            break;
            
        if error % 100 == 0:
            print('err.....')    
        
        error +=1
        sleep(ex_time)
        
        


print("Downloading Done.")



# Reduce Orderbook length and apply absolute function
matrix = np.array( reduce_orderbook(orderbooks,4) )
print('data shape: {}'.format(matrix.shape))



# normalize
abs_matrix = np.absolute(matrix)
norm_matrix = abs_matrix / abs_matrix.max()
# other ways: 
#    xmax, xmin = x.max(), x.min(),x = (x - xmin)/(xmax - xmin)
#    norm = normalize(arr)



# Write orderbook to file
df = pd.DataFrame(norm_matrix)
df.to_csv('h2/orderbooksASD_{}.csv'.format(sleep_time), index=False)

# Write price to file
df = pd.DataFrame({"x":prices})
df.to_csv('h2/pricesASD_{}.csv'.format(sleep_time), index=False)



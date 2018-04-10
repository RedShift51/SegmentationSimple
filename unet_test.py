#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np, matplotlib.pyplot as plt, os, json, cv2

import tensorflow as tf, storage as sto, matplotlib.image as mpimg

"""================================================================="""
from keras.models import Sequential
from keras.layers import Reshape
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
#from keras.regularizers import ActivityRegularizer
from keras import backend as K
import keras
from itertools import product

def Unet (nClasses=6 , input_width=320, input_height=180 , \
          nChannels=3 ): 
    
    inputs = Input((input_height, input_width, nChannels))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=-1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=-1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    
    conv6 = Convolution2D(nClasses, 1, 1, border_mode='same')(conv5)
    #conv6 = core.Reshape((nClasses,input_height*input_width))(conv6)
    #conv6 = core.Permute((2,1))(conv6)


    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    #if not optimizer is None:
	#    model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
	
    return model

keras_model = Unet()
keras_model = keras.models.load_model('keras_model_fin.h5')

""" names """
names = ['skl.031.003.left.000070.png', \
         'skl.007.019.left.000050.png', \
         'skl.023.019.left.000070.png', \
         'skl.037.002.left.000110.png', \
         'skl.006.015.left.000030.png', \
         'skl.006.002.left.000050.png', \
         'skl.010.009.left.000060.png', \
         'skl.031.009.left.000120.png', \
         'skl.007.010.left.000030.png', \
         'skl.008.021.left.000030.png']

inputs = Input((180, 320, 3))
conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[1].get_weights())(inputs)
conv1 = Dropout(0.2)(conv1)
conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[3].get_weights())(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
   
conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[5].get_weights())(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[7].get_weights())(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
 
conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[9].get_weights())(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[11].get_weights())(conv3)

up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=-1)
conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[14].get_weights())(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[16].get_weights())(conv4)
    
up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=-1)
conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[19].get_weights())(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', \
                      weights = keras_model.layers[21].get_weights())(conv5)
    
conv6 = Convolution2D(6, 1, 1, activation='relu',border_mode='same', \
                      weights = keras_model.layers[22].get_weights())(conv5)
#conv6 = core.Reshape((nClasses,input_height*input_width))(conv6)
#conv6 = core.Permute((2,1))(conv6)

conv7 = core.Activation('softmax')(conv6)

inference = Model(input=inputs, output=conv7)
opt=Adam()
inference.compile(loss="categorical_crossentropy", optimizer= opt, metrics=['accuracy'])

accs = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]}
for i0,i in enumerate(names):
    image = cv2.imread('/home/alex/Desktop/CT/skl_src/'+i)
    mask = cv2.imread('/home/alex/Desktop/CT/skl_gt/'+i)[:,:,0]
    image = cv2.resize(image, dsize = (320, 180))/255
    mask = cv2.resize(mask, dsize = (320, 180)).astype(int).astype(float)
    
    ans_inf = inference.predict(np.expand_dims(image,0))

    ans = np.argmax(ans_inf[0,:,:,:],-1)
    X = np.zeros(image.shape)
    zero = np.where(ans[:,:]==0.)
    X[[zero[0], zero[1], 0]] = 150.
    X[[zero[0], zero[1], 1]] = 150.
    X[[zero[0], zero[1], 2]] = 150.


    one = np.where(ans[:,:]==1.)
    X[[one[0], one[1], 0]] = 255.
    #X[[one[0], one[1], 2]] = 125.

    two = np.where(ans[:,:]==2.)
    #X[[two[0], two[1], 2]] = 255.
    X[[two[0], two[1], 1]] = 255.

    three = np.where(ans[:,:]==3.)
    #X[[three[0], three[1], 0]] = 150.
    #X[[three[0], three[1], 1]] = 80.
    X[[three[0], three[1], 2]] = 255.

    four = np.where(ans[:,:]==4.)
    X[[four[0], four[1], 0]] = 225.
    X[[four[0], four[1], 1]] = 225.

    five = np.where(ans[:,:]==5.)
    X[[five[0], five[1], 1]] = 255.
    X[[five[0], five[1], 2]] = 255.

    X = X/255

    added = cv2.addWeighted(image.astype(float), 0.25, X.astype(float), 0.2, 0)
    cv2.imwrite('unet_results/res'+i, added*255)
    
    """ accuracy per classes """
    accs['0'].append(np.sum((ans==0) & (mask==0))/np.max([np.sum(mask==0),1]))
    accs['1'].append(np.sum((ans==1) & (mask==1))/np.max([np.sum(mask==1),1]))
    accs['2'].append(np.sum((ans==2) & (mask==2))/np.max([np.sum(mask==2),1]))
    accs['3'].append(np.sum((ans==3) & (mask==3))/np.max([np.sum(mask==3),1]))
    accs['4'].append(np.sum((ans==4) & (mask==4))/np.max([np.sum(mask==4),1]))
    accs['5'].append(np.sum((ans==5) & (mask==5))/np.max([np.sum(mask==5),1]))
    
    
    print(i0,'of',len(names),'completed')

json.dump({'acc':accs}, open('unet_res.json','w'))




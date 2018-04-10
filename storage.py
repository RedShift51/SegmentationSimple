#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:25:43 2017

@author: alex
"""

import tensorflow as tf, numpy as np, cv2, os, matplotlib.pyplot as plt
import pandas as pd

class ConvLayer2D():
    def __init__(self, scope, kernel_shape = [2,2,3,3], num_filters = 1, \
                 padding = 'SAME', strides = [1,1,1,1], nonlin = 'relu', \
                 trainable = True, isbias = True, mode = 'conv'):
        self.weights = []
        self.mode = mode
        self.padding = padding
        self.nonlin = nonlin
        self.strides = strides
        self.isbias = isbias
        with tf.variable_scope(scope):
            for i in range(num_filters):
                with tf.variable_scope(str(i)):
                    if mode == 'deconv':
                        ks = kernel_shape
                        ks = [ks[0], ks[1], ks[3], ks[2]]
                        if self.isbias == True:
                            self.weights.append(\
                            [tf.get_variable('weights', shape = ks, \
                            initializer = tf.contrib.layers.xavier_initializer(), \
                            trainable = trainable), \
                            tf.get_variable('bias', shape = kernel_shape[-1], \
                            initializer = tf.zeros_initializer(), \
                            trainable = trainable)])
                        else:
                            self.weights.append(\
                            [tf.get_variable('weights', shape = ks, \
                            initializer = tf.contrib.layers.xavier_initializer(), \
                            trainable = trainable)])
                    else:
                        if self.isbias == True:
                            self.weights.append(\
                            [tf.get_variable('weights', shape = kernel_shape, \
                            initializer = tf.contrib.layers.xavier_initializer(), \
                            trainable = trainable), \
                            tf.get_variable('bias', shape = kernel_shape[-1], \
                            initializer = tf.zeros_initializer(), \
                            trainable = trainable)])
                        else:
                            self.weights.append(\
                            [tf.get_variable('weights', shape = kernel_shape, \
                            initializer = tf.contrib.layers.xavier_initializer(), \
                            trainable = trainable)])
    
    def transform(self, x):
        if self.mode == 'deconv':
            x1, newx, newy = x
            s = x1.get_shape().as_list()
            bs = s[0]
            out_shape = [bs,newx, newy, 1]
            #out_shape = [4,newx,newy,1]
            if self.isbias == True:
                ans = tf.concat([tf.nn.conv2d_transpose(x1, self.weights[i][0], \
                    output_shape = out_shape, padding = self.padding, \
                        strides = self.strides) + self.weights[i][1] \
                        for i in range(len(self.weights))], axis = 3)
            else:
                ans = tf.concat([tf.nn.conv2d_transpose(x1, self.weights[i][0], \
                    output_shape = out_shape, padding = self.padding, \
                        strides = self.strides) \
                        for i in range(len(self.weights))], axis = 3)
        else:    
            if self.isbias == True:
                ans = tf.concat([tf.nn.conv2d(x, self.weights[i][0], \
                                              padding = self.padding, \
                        strides = self.strides) + self.weights[i][1] \
                        for i in range(len(self.weights))], axis = 3)
            else:
                ans = tf.concat([tf.nn.conv2d(x, self.weights[i][0], \
                                              padding = self.padding, \
                        strides = self.strides) \
                        for i in range(len(self.weights))], axis = 3)
    
        if self.nonlin == 'relu':
            return tf.nn.relu(ans)
        elif self.nonlin == 'sigmoid':
            return tf.nn.sigmoid(ans)
        else:
            return ans

    def __call__(self, x):
        return self.transform(x)

def convblock(X, scope, trainable = True):
    with tf.variable_scope(scope):
        num_dim = X.get_shape().as_list()[3]
        X = ConvLayer2D('conv1', kernel_shape = [4,4,num_dim,1], num_filters = 64, \
                strides = [1,2,2,1], trainable = trainable)(X)
        X = ConvLayer2D('conv2', kernel_shape = [4,4,64,1], num_filters = 64, \
                strides = [1,2,2,1], trainable = trainable)(X)
        X = tf.nn.max_pool(X, ksize = [1,4,4,1], strides = [1,1,1,1], \
                    padding = 'SAME')
        X = tf.contrib.layers.batch_norm(X, trainable = trainable)
        return X

def upsampblock(X, scope, outdims=7):
    with tf.variable_scope(scope):
        num_dim = X.get_shape().as_list()[3]
        s1 = X.get_shape().as_list()
        #print(s1)
        X = tf.image.resize_nearest_neighbor(X, [int(s1[1]*23/12), s1[2]*2])
        X = ConvLayer2D('conv1', kernel_shape = [4,4,num_dim,1], num_filters = 32)(X)
        s2 = X.get_shape().as_list()
        #print(s2)
        X = tf.image.resize_nearest_neighbor(X, [int(s2[1]*45/23), s2[2]*2])
        X = ConvLayer2D('conv2', kernel_shape = [4,4,32,1], num_filters = 32)(X)
        """ differences are these 4 strings and 64 filters (were 32) """
        #X = tf.contrib.layers.batch_norm(X)
        #X = tf.nn.relu(X)
        #X = tf.contrib.layers.batch_norm(X)
        #X = tf.nn.relu(X)
        
        s3 = X.get_shape().as_list()
        #print(s3)
        X = tf.image.resize_nearest_neighbor(X, [s3[1]*2, s3[2]*2])
        X = ConvLayer2D('conv3', kernel_shape = [4,4,32,1], num_filters = 32)(X)
        s4 = X.get_shape().as_list()
        #print(s4)
        X = tf.image.resize_nearest_neighbor(X, [s4[1]*2, s4[2]*2])
        X = ConvLayer2D('conv4', kernel_shape = [4,4,32,1], num_filters = outdims)(X)
        return X
"""
def upsampblock1(X, scope, outdims=7):
    with tf.variable_scope(scope):
        num_dim = X.get_shape().as_list()[3]
        s1 = X.get_shape().as_list()
        #print(s1)
        X = tf.image.resize_nearest_neighbor(X, [int(s1[1]*23/12), s1[2]*2])
        X = ConvLayer2D('conv1', kernel_shape = [4,4,num_dim,1], num_filters = 32)(X)
        X = tf.contrib.layers.batch_norm(X)
        X1 = tf.nn.relu(X)
        X1 = tf.contrib.layers.batch_norm(X1)
        X1 = tf.nn.relu(X1)
        X = X1 + X
        
        s2 = X.get_shape().as_list()
        #print(s2)
        X = tf.image.resize_nearest_neighbor(X, [int(s2[1]*45/23), s2[2]*2])
        X = ConvLayer2D('conv2', kernel_shape = [4,4,32,1], num_filters = 32)(X)
        
        X = tf.contrib.layers.batch_norm(X)
        X1 = tf.nn.relu(X)
        X1 = tf.contrib.layers.batch_norm(X1)
        X1 = tf.nn.relu(X1)
        X = X1 + X
        
        #X = tf.contrib.layers.batch_norm(X)
        #X = tf.nn.relu(X)
        #X = tf.contrib.layers.batch_norm(X)
        #X = tf.nn.relu(X)
        
        s3 = X.get_shape().as_list()
        #print(s3)
        X = tf.image.resize_nearest_neighbor(X, [s3[1]*2, s3[2]*2])
        X = ConvLayer2D('conv3', kernel_shape = [4,4,32,1], num_filters = 32)(X)
        X = tf.contrib.layers.batch_norm(X)
        X1 = tf.nn.relu(X)
        X1 = tf.contrib.layers.batch_norm(X1)
        X1 = tf.nn.relu(X1)
        X = X1 + X
        
        s4 = X.get_shape().as_list()
        #print(s4)
        X = tf.image.resize_nearest_neighbor(X, [s4[1]*2, s4[2]*2])
        X = ConvLayer2D('conv4', kernel_shape = [4,4,32,1], num_filters = outdims)(X)
        
        return X
"""
"""==============================================================="""
def inception(X, scope):
    with tf.variable_scope(scope):
        s1 = X.get_shape().as_list()
        X1 = ConvLayer2D('conv1', kernel_shape = [5,5,s1[-1],1], num_filters = 8, \
                        nonlin = 'Nan', isbias = False)(X)
        X2 = ConvLayer2D('conv2', kernel_shape = [3,3,s1[-1],1], num_filters = 8, \
                        nonlin = 'Nan', isbias = False)(X)
        X3 = ConvLayer2D('conv3', kernel_shape = [1,1,s1[-1],1], num_filters = 8, \
                        nonlin = 'Nan', isbias = False)(X)
        X4 = tf.nn.max_pool(X3, ksize = [1,4,4,1], strides = [1,1,1,1], \
                    padding = 'SAME')
        X = tf.concat([X1,X2,X3,X4], axis = -1)
        X = tf.nn.relu(X)
        return X

def resblock(X, scope):
    with tf.variable_scope(scope):
        
        X = tf.contrib.layers.batch_norm(X)
        
        X = inception(X, 'inc1')
        X1 = tf.nn.elu(X)
        X1 = tf.contrib.layers.batch_norm(X1)
        X1 = inception(X1, 'inc2')
        X = X1 + X
        return X

def upsampblock01(X, scope, outdims=7):
    with tf.variable_scope(scope):
        s1 = X.get_shape().as_list()
        X = tf.image.resize_nearest_neighbor(X, [int(s1[1]*2), s1[2]*2])#23/12
        X = resblock(X, 'res1')
        
        s2 = X.get_shape().as_list()
        X = tf.image.resize_nearest_neighbor(X, [int(s2[1]*2), s2[2]*2])#45/23
        X = resblock(X, 'res2')
        
        s3 = X.get_shape().as_list()
        X = tf.image.resize_nearest_neighbor(X, [int(s3[1]*272/270*2), s3[2]*2])
        X = resblock(X, 'res3')
        
        s4 = X.get_shape().as_list()
        X = tf.image.resize_nearest_neighbor(X, [s4[1]*2, s4[2]*2])
        X = ConvLayer2D('conv', kernel_shape = [1,1,s4[-1],1], num_filters = outdims, \
                        nonlin = 'Nan')(X)
        return X
        

"""==============================================================="""
def upsampblock1(X, scope, outdims=7, numfshapes = 16):
    with tf.variable_scope(scope):
        s1 = X.get_shape().as_list()
        print(s1)
        X = ConvLayer2D('dec1', kernel_shape = [4,4,s1[-1],1], num_filters = 32, \
                        strides = [1,2,2,1], mode='deconv')([X, int(s1[1]*2), s1[2]*2])
        X = resblock(X, 'res1')
        s2 = X.get_shape().as_list()
        print(s2)
        X = ConvLayer2D('dec2', kernel_shape = [4,4,s2[-1],1], num_filters = 32, \
                        strides = [1,2,2,1], mode='deconv')([X, int(s2[1]*2*270/272), s2[2]*2])
        X = resblock(X, 'res2')
        s3 = X.get_shape().as_list()
        print(s3)
        X = ConvLayer2D('dec3', kernel_shape = [4,4,s3[-1],1], num_filters = 32, \
                        strides = [1,2,2,1], mode='deconv')([X, int(s3[1]*2), s3[2]*2])
        X = resblock(X, 'res3')
        s4 = X.get_shape().as_list()
        print(s4)
        X = ConvLayer2D('dec4', kernel_shape = [4,4,s4[-1],1], num_filters = numfshapes, \
                        strides = [1,2,2,1], mode='deconv')([X, int(s4[1]*2), s4[2]*2])
        s4 = X.get_shape().as_list()
        print(s4)
        X = ConvLayer2D('conv', kernel_shape = [1,1,s4[-1],1], num_filters = outdims, \
                        nonlin = 'Nan')(X)
        return X

"""==============================================================="""
"""==============================================================="""
"""
def inception(X, scope):
    with tf.variable_scope(scope):
        s1 = X.get_shape().as_list()
        X1 = ConvLayer2D('conv1', kernel_shape = [5,5,s1[-1],1], num_filters = 22, \
                        nonlin = 'Nan', isbias = False)(X)
        X2 = ConvLayer2D('conv2', kernel_shape = [3,3,64,1], num_filters = 21, \
                        nonlin = 'Nan', isbias = False)(X)
        X3 = ConvLayer2D('conv3', kernel_shape = [1,1,64,1], num_filters = 21, \
                        nonlin = 'Nan', isbias = False)(X)
        X = tf.concat([X1,X2,X3], axis = -1)
        X = tf.nn.relu(tf.contrib.layers.batch_norm(X))
        return X
"""

def upsampblock2(X, scope, outdims = 7):
    with tf.variable_scope(scope):
        num_dim = X.get_shape().as_list()[3]
        s1 = X.get_shape().as_list()
        #print(s1)
        
        X = tf.image.resize_nearest_neighbor(X, [int(s1[1]*2), s1[2]*2])
        X = inception(X,'inc1')

        s2 = X.get_shape().as_list()
        #print(s2)
        X = tf.image.resize_nearest_neighbor(X, [int(s2[1]*1.25), s2[2]*2])
        X = inception(X, 'inc2')

        s3 = X.get_shape().as_list()
        #print(s3)
        X = tf.image.resize_nearest_neighbor(X, [s3[1]*2, s3[2]*2])
        X = inception(X, 'inc3')

        #print(s4)
        s4 = X.get_shape().as_list()
        X = tf.image.resize_nearest_neighbor(X, [s4[1]*3, s4[2]*2])
        X = inception(X, 'inc4')
        s4 = X.get_shape().as_list()
        X = ConvLayer2D('conv7', kernel_shape = [1,1,s4[-1],1], num_filters = outdims, \
                        nonlin = 'Nan', isbias = False)(X)
        
        return X


"""==============================================================="""        
def small_upsamp(X, scope):
    with tf.variable_scope(scope):
        s1 = X.get_shape().as_list()
        X = tf.image.resize_nearest_neighbor(X, [s1[1]*2, s1[2]*2])
        X = ConvLayer2D('conv1', kernel_shape = [1,1,s1[-1],1], num_filters = 16,\
                        padding = 'SAME', strides = [1,1,1,1])(X)
        #print(s1)
        s1 = X.get_shape().as_list()
        X = tf.image.resize_nearest_neighbor(X, [s1[1]*2, s1[2]*2])
        X = ConvLayer2D('conv3', kernel_shape = [1,1,16,1], num_filters = 7,\
                        padding = 'SAME', strides = [1,1,1,1])(X)
        #print(s1)
        return X

def end2end(X, scope, batchsize):
    with tf.variable_scope(scope):
        batchsize = batchsize
        X = tf.slice(X, [0,0,0,0], [batchsize, -1,-1,-1])
        s1 = X.get_shape().as_list()
        print(s1)
        X1 = resblock(X, 'res1')
        X2 = tf.nn.max_pool(X1, ksize = [1,4,4,1], strides = [1,2,2,1], \
                    padding = 'SAME')
        s2 = X2.get_shape().as_list()
        print(s2)
        X2 = resblock(X2, 'res2')
        X3 = tf.nn.max_pool(X2, ksize = [1,4,4,1], strides = [1,2,2,1], \
                    padding = 'SAME')
        s3 = X3.get_shape().as_list()
        print(s3)
        X3 = resblock(X3, 'res3')
        X3 = ConvLayer2D('dec1', kernel_shape = [4,4,s3[-1],1], num_filters = 32, \
            strides = [1,2,2,1], padding='SAME',mode='deconv')([X3, int(s3[1]*2), s3[2]*2])+X2
        #X3 = tf.pad(X3, [[0,0],[1,0],[0,0],[0,0]])
        #x3shape = tf.shape(X3)
        #zshape = tf.slice(x3shape,[0],[1])
        #X3 = tf.concat([X3, tf.zeros([zshape,1,s3[2],s3[3]])],axis=1)#tf.zeros([s3[0],1,s3[2],s3[3]])], axis=1)
        #X3 = X3 + tf.slice(X2,[0,0,0,0],[-1,89,-1,-1])
        #X3 = tf.pad(X3, [[0,0],[1,0],[0,0],[0,0]])
        s4 = X3.get_shape().as_list()
        print(s4)
        X3 = resblock(X3, 'res4')
        X3 = ConvLayer2D('dec2', kernel_shape = [4,4,s4[-1],1], num_filters = 32, \
            strides = [1,2,2,1], mode='deconv')([X3, int(s4[1]*2), s4[2]*2])+X1
        #X3 = 
        #X3 = X3 + X1
        s5 = X3.get_shape().as_list()
        print(s5)
        X3 = ConvLayer2D('conv3', kernel_shape = [1,1,s5[-1],1], num_filters = 6,\
                        nonlin = 'Nan',padding = 'SAME', strides = [1,1,1,1])(X3)
        return X3
        
        

def encoder(X, scope, trainable = True):
    with tf.variable_scope(scope):
        X = convblock(X, 'encblock1', trainable=trainable)
        X = convblock(X, 'encblock2', trainable=trainable)
        return X

def decoder(X, scope, outdims = 7, numfinshapes=16):
    with tf.variable_scope(scope):
        X = upsampblock1(X, 'decblock1', outdims, numfinshapes)
        return X



def wcrossentropy(Y,X,counts):
    counts_arr = tf.constant(np.array([float(counts[i]) for i in list(counts.keys())]))
    counts_arr = tf.expand_dims(counts_arr,0)
    counts_arr = tf.expand_dims(counts_arr,0)
    counts_arr = tf.expand_dims(counts_arr,0)
    
    return -tf.reduce_mean(tf.multiply(tf.multiply(tf.log(\
                tf.clip_by_value(tf.nn.softmax(X),0.00001,0.99999)), \
                tf.cast(counts_arr, tf.float32)), tf.cast(Y, tf.float32)))

#inc_block(X0, 't')
""" Sequence """
""" Вначале делаем маски, выравниваем ravel'ом все, потом оставляем
только те столбцы, где значение Y > 0 """

def normalized_loss(X, Y):
    Yaux = Y#tf.reshape(Y, [180*320, 7])
    Xaux = X#tf.reshape(X, [180*320, 7])
    #aux_var = Y.get_shape().as_list()[0]
    #print(tf.slice(Y, [0,0,0,0], \
    #            [-1,-1,-1,1]).get_shape().as_list())
    layer_mask = lambda i : tf.slice(Y, [0,0,0,i], \
                [-1,-1,-1,1])#, [aux_var, 180*320])
    
    create_bool = lambda i : tf.not_equal(layer_mask(i),0)
    
    compact_X = lambda i : tf.boolean_mask(Xaux, tf.squeeze(create_bool(i),-1))
    compact_Y = lambda i : tf.boolean_mask(Yaux, tf.squeeze(create_bool(i),-1))
    count_loss = lambda i : tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                        labels = compact_Y(i), logits = compact_X(i)))
    aux_arr = tf.range(Y.get_shape().as_list()[-1])
    #normalized = tf.constant([1., 1.5, 3., 3., 1., 1., 1.], dtype = tf.float32)
    res = tf.map_fn(count_loss, aux_arr, dtype = tf.float32)
    #res = tf.multiply(res, normalized)
    return tf.reduce_mean(res)

def onehot(Xmask):
    X = np.zeros(shape = (Xmask.shape[0],Xmask.shape[1],7))
    for i in range(len(Xmask[:,0])):
        for j in range(len(Xmask[0,:])):
            X[i,j,Xmask[i,j]] = 1.
    return X

""" 180 320   -   1080 1920 """
def augment(Xname, num):
    X, Xmask = cv2.imread('/home/alex/Desktop/CT/skl_src/'+Xname), \
                cv2.imread('/home/alex/Desktop/CT/skl_gt/'+Xname)
    if max(X.ravel()) > 2:
        X = X * 1. / 255
    for i in range(num):
        if i == 0:
            Xres = cv2.resize(X, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
            Xmaskres = cv2.resize(Xmask, dsize = (320, 180))#[:,:,0:1]#, \
                                  #interpolation=cv2.INTER_AREA)[:,:,0:1]
            Xmaskres = onehot(Xmaskres)
        elif (i > 5) and (i < 10):
            randposx = np.random.randint(low=0,high=int(1920/2))
            randposy = np.random.randint(low=200,high=int(1080/2))
            Xres = X[randposy:randposy+540,randposx:randposx+960,:]
            Xmaskres = Xmask[randposy:randposy+540,randposx:randposx+960,0:1]
            
            Xres = cv2.resize(Xres, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
            Xmaskres = cv2.resize(Xmaskres, dsize = (320, 180))#[:,:,0:1]#, \
            Xmaskres = onehot(Xmaskres)
        elif (i > 0) and (i < 6):
            randposx = np.random.randint(low=0,high=int(1920/4))
            randposy = np.random.randint(low=200,high=int(1080/4))
            Xres = X[randposy:randposy+810,randposx:randposx+1440,:]
            Xmaskres = Xmask[randposy:randposy+810,randposx:randposx+1440,0:1]
            
            Xres = cv2.resize(Xres, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
            Xmaskres = cv2.resize(Xmaskres, dsize = (320, 180))#[:,:,0:1]#, \
            Xmaskres = onehot(Xmaskres)
        else:
            break
        yield Xres, Xmaskres


def augment_one(Xname):
    for i in Xname:
        X, Xmask = cv2.imread('/home/alex/Desktop/CT/skl_src/'+i), \
                cv2.imread('/home/alex/Desktop/CT/skl_gt/'+i)
        X = cv2.resize(X, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
        Xmask = cv2.resize(Xmask, dsize = (320, 180))#[:,:,0:1]#, \
        Xmask = onehot(Xmask)
        yield X, Xmask

def augment_bin(Xname, num=3, clsnum=0):
    X, Xmask = cv2.imread('/home/alex/Desktop/CT/skl_src/'+Xname)/255., \
                cv2.imread('/home/alex/Desktop/CT/skl_gt/'+Xname)
    Xmask = Xmask[:,:,0:1]
    pos = np.where(Xmask == clsnum)
    Xmask = np.zeros(Xmask.shape).astype(float)
    Xmask[pos] = 1.
    if max(X.ravel()) > 2:
        X = X * 1. / 255
    for i in range(num):
        if i == 0:
            Xres = cv2.resize(X, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
            Xmaskres = cv2.resize(Xmask, dsize = (320, 180))
            pos = np.where(Xmaskres >= 0.5)
            Xmaskres = np.zeros(Xmaskres.shape).astype(float)
            Xmaskres[pos] = 1.
            Xmaskres = np.expand_dims(Xmaskres, axis=-1)
        elif (i > 0) and (i < 6):
            randposx = np.random.randint(low=0,high=int(1920/2))
            randposy = np.random.randint(low=200,high=int(1080/2))
            Xres = X[randposy:randposy+540,randposx:randposx+960,:]
            Xmaskres = Xmask[randposy:randposy+540,randposx:randposx+960,0:1]
            
            Xres = cv2.resize(Xres, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
            Xmaskres = cv2.resize(Xmaskres, dsize = (320, 180))
            
            pos = np.where(Xmaskres >= 0.5)
            Xmaskres = np.zeros(Xmaskres.shape).astype(float)
            Xmaskres[pos] = 1.
            Xmaskres = np.expand_dims(Xmaskres, axis=-1)
            
        else:
            break
        
        yield Xres, Xmaskres
 
def augment_bin_sc(Xname, num=3, clsnum=2):
    X, Xmask = cv2.imread('/home/alex/Desktop/CT/skl_src/'+Xname), \
                cv2.imread('/home/alex/Desktop/CT/skl_gt/'+Xname)
    Xmask = Xmask[:,:,0:1]
    pos = np.where(Xmask == clsnum)
    Xmask = np.zeros(Xmask.shape).astype(float)
    Xmask[pos] = 1.
    for i in range(num):
        randposx = np.random.randint(low=0,high=int(1920/2))
        Xres = X[540:,randposx:randposx+960,:]
        Xmaskres = Xmask[540:,randposx:randposx+960,0:1]
        Xres = cv2.resize(Xres, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
        Xmaskres = cv2.resize(Xmaskres, dsize = (320, 180))
        yield Xres, np.expand_dims(Xmaskres,-1)            
    
"""
for i0,i in enumerate(train_names):
    Xmask = cv2.imread('/home/alex/Desktop/CT/skl_gt/'+i)
    Xmask = Xmask[:,:,0:1]
"""    
""" 65x65 patches """
def augm_2_approach(Xname, num = 4):
    X, Xmask = cv2.imread('/home/alex/Desktop/CT/skl_src/'+Xname), \
                cv2.imread('/home/alex/Desktop/CT/skl_gt/'+Xname)
    Xmask = Xmask[:,:,0:1]
    """ Padded pictures """
    Xmaskpad = cv2.copyMakeBorder(Xmask,32,32,32,32,cv2.BORDER_CONSTANT)
    Xpad = cv2.copyMakeBorder(X,32,32,32,32,cv2.BORDER_CONSTANT)
    list_cls = pd.unique(Xmask.ravel())
    for i0,i in enumerate(list_cls):
        poses = np.where(Xmask == i)
        for j in range(num):
            n = np.random.randint(low=0,high=len(poses[0]))
            ans = np.zeros(7)
            ans[Xmaskpad[poses[0][n]+33, poses[1][n]+33]] = 1.
            yield Xpad[poses[0][n]:poses[0][n]+65,poses[1][n]:poses[1][n]+65,:]/255., \
                ans
            
""" 2,3,4,5,6 """
def augm_small(Xname, num = 4):
    X, Xmask = cv2.imread('/home/alex/Desktop/CT/skl_src/'+Xname)*1./255, \
                cv2.imread('/home/alex/Desktop/CT/skl_gt/'+Xname)
    Xmask = Xmask[:,:,0:1]
    #pos = np.where(Xmask == clsnum)
    #Xmask = np.zeros(Xmask.shape).astype(float)
    #Xmask[pos] = 1.
    #if sum(Xmask.ravel() == 1.) < 0.0005 * 1080* 1920:
    #    return []
    uni = np.sort(np.unique(Xmask.ravel()))
    if uni[-1] == 6:
        uni = uni[:-1]
    uni1 = uni[1:]
    np.random.shuffle(uni1)     
    uni = list(uni)+list(uni1)
    for i in uni:
        print(i,'th img is processing of', len(uni))
        yes, iternum = 0, 0
        if np.random.rand() > 0.15:
            while yes == 0:
                randy, randx = np.random.randint(0,int(1080*3/4)), \
                        np.random.randint(400,int(1920*3/4))
                Xres = X[randy:randy+180,randx:randx+320,:]
                Xmaskres = Xmask[randy:randy+180,randx:randx+320,0:1]
                if np.sum(Xmaskres==i)/320/180 > 0.15:
                    yes = 1
                else:
                    iternum += 1
                    print(iternum)
                    if iternum > 60:
                        break
                    continue
                
                Xmaskres = onehot(Xmaskres)[:,:,:-1]
                yield Xres, Xmaskres
            #Xres = cv2.resize(Xres, dsize = (160, 90), interpolation=cv2.INTER_CUBIC)
            #Xmaskres = cv2.resize(Xmaskres, dsize = (160, 90))
        else:
            while yes == 0:
                randy, randx = np.random.randint(0,int(1080*2/3)), \
                        np.random.randint(400,int(1920*2/3))
                Xres = X[randy:randy+360,randx:randx+640,:]
                Xmaskres = Xmask[randy:randy+360,randx:randx+640,0:1]
                Xres = cv2.resize(Xres, dsize = (320, 180), interpolation=cv2.INTER_CUBIC)
                Xmaskres = cv2.resize(Xmaskres, dsize = (320, 180)).astype(int)
                Xmaskres = np.expand_dims(Xmaskres, -1)
                if np.sum(Xmaskres==i)/320/180 > 0.15:
                    yes = 1
                else:
                    iternum += 1
                    print(iternum)
                    if iternum > 80:
                        break    
                    continue
                Xmaskres = onehot(Xmaskres)[:,:,:-1]
                yield Xres, Xmaskres
    
    
def multiform(arr):
    uni = np.array([i for i in pd.unique(arr.ravel()) if i != 6])
    nums = [np.sum(arr==uni[i])/np.size(arr) for i in range(len(uni))]
    return -np.sum([i*np.log2(i) for i in nums])

def augm_small1(X, Y):
    #uni = np.array([i for i in pd.unique(Y.ravel()) if i != 6])  
    order = pd.DataFrame({'order' : np.arange(len(Y[:,0,0,0])), \
                'meas' : [multiform(Y[i,:,:,:]) for i in range(len(Y[:,0,0,0]))]})
    order = order.sort_values('meas', 0, False).reset_index(drop=True)
    patches, level = pd.DataFrame({'Ynum':[],'x':[],'y':[],'meas':[]}), \
                        np.arange(40, 5, -10)
    for i0,i in enumerate(order['order']):
        arr = Y[i]
        
        patchList, coords, meas = [], [], []
        for j in range(80):
            randy, randx = np.random.randint(0,int(1080*3/4)), \
                        np.random.randint(400,int(1920*3/4))
            patchList.append(arr[randy:randy+180,randx:randx+320])    
            coords.append([randy, randx])
            meas.append(multiform(arr[randy:randy+180,randx:randx+320]))
        
        res_i = pd.DataFrame({'meas':meas, 'x':[k[1] for k in coords], \
                              'y':[k[0] for k in coords], 'Ynum':i})
        res_i = res_i.sort_values('meas', 0, False).reset_index(drop=True)
        res_i = res_i[:level[i0]]
        
        patches = patches.append(res_i, ignore_index = True)
    patches = patches.sort_values('meas', 0, False).reset_index(drop=True)
    patches = patches[:6]    
    for i in range(len(patches)):
        yield X[int(patches['Ynum'][i]), int(patches['y'][i]):int(patches['y'][i])+180, \
                    int(patches['x'][i]):int(patches['x'][i])+320,:], \
        onehot(Y[int(patches['Ynum'][i]), int(patches['y'][i]):int(patches['y'][i])+180, \
                    int(patches['x'][i]):int(patches['x'][i])+320,0])[:,:,:-1]



#plt.imshow(Y[int(patches['Ynum'][0]), int(patches['y'][0]):int(patches['y'][0])+180, int(patches['x'][0]):int(patches['x'][0])+320,0]) 





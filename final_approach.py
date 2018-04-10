#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt, os, json, cv2

import tensorflow as tf, storage as sto, matplotlib.image as mpimg

names = os.listdir('/home/alex/Desktop/CT/skl_src/')
train_names = np.arange(len(names))
np.random.shuffle(train_names)
valid_names = list(np.array(names)[train_names[352:]])
train_names = list(np.array(names)[train_names[:352]])

""" Firstly let's mine frequences """
"""
counts = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
for i0,i in enumerate(names):
    pict = cv2.imread('/home/alex/Desktop/CT/skl_gt/'+i)[:,:,0] 
    counts['0'] += np.sum(pict==0)
    counts['1'] += np.sum(pict==1)
    counts['2'] += np.sum(pict==2)
    counts['3'] += np.sum(pict==3)
    counts['4'] += np.sum(pict==4)
    counts['5'] += np.sum(pict==5)
    print(i0,'out of',len(names))
    
counts = {i:180*320*(i0+1)/counts[i] for i in list(counts.keys())}
"""
tf.reset_default_graph()
X0 = tf.placeholder(dtype=tf.float32, shape=[None,180,320,3])
Y0 = tf.placeholder(dtype=tf.float32, shape = [None,180,320,6])

#X = sto.encoder(X0, 'encoder')
#X = sto.decoder(X, 'decoder', outdims=3)
#X = tf.nn.sigmoid(X)
X = sto.end2end(X0, '1', 4)
#Xval = sto.end2end(X0, '1', 2)

vars = tf.trainable_variables()
#m = 0.0001
loss = sto.wcrossentropy(Y0, X, counts) + \
        tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.003
#tf.nn.softmax_cross_entropy_with_logits(labels=Y0,logits=X) + \
#        tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.003

optimizer = tf.train.AdamOptimizer().minimize(loss)
opt_small = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)


acc = tf.contrib.metrics.accuracy(predictions=tf.argmax(tf.nn.softmax(X), axis=-1),\
                                  labels=tf.argmax(Y0, axis=-1))

"""
XY = [list(sto.augm_small1(i, 1))[0] for i in valid_names[:5]]
Xval = np.concatenate([np.expand_dims(j[0],0) for j in XY],0)
if max(Xval.ravel()) > 2:
    Xval = Xval * 1. / 255
del XY
"""
saver = tf.train.Saver()

#cv2.imwrite('/home/alex/Desktop/CT/approach2/final_appr_data/real'+'.png', 255*Xval[0,:,:,:])
loss_tr, loss_val, acc_tr, acc_val = [], [], [], []
#batch = [train_names[4*i:4*i+4] for i in range(int(len(names)/4))]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, '/home/alex/Desktop/CT/approach2/finmodel')
    for e in range(5):
        batch = [train_names[4*i:4*i+4] for i in range(int(len(names)/4))]  
        for i0,i in enumerate(batch):
            #i0 = i01+31
            Xz = [np.expand_dims(cv2.resize(cv2.imread('/home/alex/Desktop/CT/skl_src/'+j),(320,180)),\
                                 0)*1./255 for j in i]
            Xz = np.concatenate(Xz, 0)
        
            Yz = [np.expand_dims(sto.onehot(\
                            cv2.resize(cv2.imread('/home/alex/Desktop/CT/skl_gt/'+j),(320,180))\
                            [:,:,0].astype(int)), 0)[:,:,:,:-1] for j in i]
            #Yz = [np.expand_dims(cv2.imresize(cv2.imread('/home/alex/Desktop/CT/skl_gt/'+j),(320,180)).astype(int).astype(float),\
            #        0)[:,:,:,:1] for j in i]
            Yz = np.concatenate(Yz, 0)
        
            #XY = list(sto.augm_small1(Xz, Yz))        
        
            #Xz = np.concatenate([np.expand_dims(j[0],0) for j in XY])
            #Yz = np.concatenate([np.expand_dims(j[1],0) for j in XY])
        
        
            #for j in range(10):
            if i0 >= 40:
                _,a, xres, b = sess.run([opt_small, loss, X, acc], \
                            feed_dict = {X0:Xz[:4,:,:,:], Y0:Yz[:4,:,:,:]})
            else:    
                _,a, xres, b = sess.run([optimizer, loss, X, acc], \
                            feed_dict = {X0:Xz[:4,:,:,:], Y0:Yz[:4,:,:,:]})
            loss_tr += [float(np.mean(a))]
            acc_tr += [float(b)]
            
            
            a, ans1, b = sess.run([loss, X, acc], feed_dict = \
                        {X0:np.concatenate([Xz[-2:,:,:,:], Xz[-2:,:,:,:]],0), \
                         Y0:np.concatenate([Yz[-2:,:,:,:], Yz[-2:,:,:,:]],0)})    
            loss_val += [float(np.mean(a))]
            acc_val += [float(b)]
            
            
            #cv2.imwrite('/home/alex/Desktop/CT/approach2/final_appr_data/val'\
            #            +str(i0)+'.png', 255*ans[0])
            print('===================================================')
            print('Batch num', i0, 'epo num', e)
            print('Loss', loss_tr[-1], loss_val[-1])
            print('Acc', acc_tr[-1], acc_val[-1])
            #saver.save(sess, '/home/alex/Desktop/CT/approach2/encoder')
            if (i0+1)%5 == 0:
                for k in range(len(xres[:,0,0,0])):
                    cv2.imwrite('/home/alex/Desktop/CT/approach2/final_appr_data/train/'+\
                    str(i0)+str(k)+'out.png', np.argmax(xres[k,:,:,:],-1)*10)
                    cv2.imwrite('/home/alex/Desktop/CT/approach2/final_appr_data/train/'+\
                    str(i0)+str(k)+'src.png', Xz[k,:,:,:]*255)
                    cv2.imwrite('/home/alex/Desktop/CT/approach2/final_appr_data/train/'+\
                    str(i0)+str(k)+'true.png', np.argmax(Yz[k,:,:,:],-1)*10)
            
            if (i0+1) % 15 == 0:
                json.dump({'loss_val':loss_val, 'loss_tr':loss_tr, \
                       'acc_val':acc_val, 'acc_tr':acc_tr}, \
                        open('/home/alex/Desktop/CT/approach2/losses_fm_CT.json', 'w'))
                saver.save(sess, '/home/alex/Desktop/CT/approach2/finmodel1')

#sess = tf.Session()
#saver.restore(sess, '/home/alex/Desktop/CT/approach2/encoder')

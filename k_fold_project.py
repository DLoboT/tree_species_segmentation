#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:35:46 2019

@author: daliana
"""

#%%
from keras.optimizers import SGD, Adam, Adamax
from keras.models import Model, load_model
from keras import applications
from keras import backend as K
import keras
import numpy as np
from sklearn.utils import shuffle
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import multiprocessing
from time import sleep
from skimage.util.shape import view_as_windows
import h5py
import cv2
import time
from sklearn import preprocessing as pp
#%%
def Directory():
    a = os.listdir("./Cumbaru/CUMBARUS_polig/")

    seq = []
    for i in range(len(a)):
        a[i] = "./Cumbaru/CUMBARUS_polig/" + a[i] + "/"

    for i in range(len(a)):
        v = os.listdir(a[i])
        for j in range(len(v)):
            v[j] = a[i] + v[j]
        seq += v

    Images = []
    for file in seq:
        if file.endswith(".JPG"):
            Images.append(file)
    
    Images = shuffle(Images, random_state = 1)
  
    return Images

#%%
def load(j):
    try:
        im = Image.open(j[: len(j) - 4] + "_groundtruth.tif")

        y_set  = np.asarray(im)
    
        im = Image.open(j)

        x_set = np.asarray(im)
        
    except FileNotFoundError:
        return [np.nan, np.nan, np.nan]
    
    return [x_set, y_set, j]

#%%
def load_im(dataset, patches_size): 
    sleep(0.5)
    num_cores = multiprocessing.cpu_count()    
    pool = multiprocessing.Pool(num_cores)
    
    im = pool.map(load, dataset, chunksize = 1)    
    
    pool.close()
    pool.join()
    
    x = []
    y = []
    files_name = []
    
    for i in range(len(im)):
        x_set, y_set, z = im[i]
        if not np.sum(np.isnan(x_set)):
            x.append(x_set)
            y.append(y_set)
            files_name.append(z)
    
    if not len(x):
        raise FileExistsError

    images = np.asarray(x).astype('float64')
    
    groundtruth = np.asarray(y)
        
    return images, groundtruth, files_name

#%% 
def patches(images, groundtruth, patches_size, overlap_percent=0):
    print(len(images))
    
    overlap = round(patches_size * overlap_percent)
    overlap -= overlap % 2
    stride = patches_size - overlap

    # define the window_shape for patch extraction
    window_shape_array = (1, patches_size, patches_size, images.shape[3])
    stride_array = (1, stride, stride, images.shape[3])
    window_shape_ref = (1, patches_size, patches_size)
    stride_ref = (1, stride, stride)
    
    # exctract patches
    patches_array = np.array(view_as_windows(images, window_shape_array, step = stride_array))
    patches_ref = np.array(view_as_windows(groundtruth, window_shape_ref, step = stride_ref))
    
    m, k1, k2, p, n, row, col, depth = patches_array.shape


    patches_array = patches_array.reshape(m*k1*k2,row,col,depth)   
    patches_ref = patches_ref.reshape(m*k1*k2,row,col) 
    print(k1*k2)
    
    return patches_array, patches_ref, k1, k2

#%% 
def rut(dataset_b, flag, patches_size):
        
        global net, Arq
        images, groundtruth, _ = load_im(dataset_b, patches_size)

        x_b, y_b,_,_ = patches(images, groundtruth, patches_size, overlap_percent)
        
        if Arq != 3:
            x_b = x_b.astype('float32') / 255
                      
        # Hot encoding 
        num_classes = len(np.unique(y_b))
        
        imgs_mask_cat = y_b.reshape(-1)

        imgs_mask_cat = keras.utils.to_categorical(imgs_mask_cat,num_classes)
        y_b_h = imgs_mask_cat.reshape(y_b.shape[0],y_b.shape[1],y_b.shape[2],num_classes) 
       #y_b_h = imgs_mask_cat.reshape(y_b.shape[0],y_b.shape[1]*y_b.shape[2],num_classes) 

        #minibatch for the different arquitectures
        #Segnet = 10
        #Unet = 16
        #fc_dense = 6
        #mobilenetv2 = 7
        #xception = 2
        if Arq == 1:
            minibatch_size = 10
        elif Arq == 2:
            minibatch_size = 16
        elif Arq == 3:
            minibatch_size = 2
        else:
            minibatch_size = 6
        

        n_minibatch = x_b.shape[0] // minibatch_size
        
        loss = np.zeros((1 , 2))
        # Training the network per minibatch
        for  minibatch in range(n_minibatch):

            x_mb = x_b[minibatch * minibatch_size : (minibatch + 1) * minibatch_size , : , : , :]
            y_h_mb = y_b_h[minibatch * minibatch_size : (minibatch + 1) * minibatch_size , : ]

            if flag:
                loss += net.train_on_batch(x_mb, y_h_mb)
            else:
                loss += net.test_on_batch(x_mb, y_h_mb)
    
        # Getting the remaining mini-batch of the training set  
        if x_b.shape[0] % minibatch_size:
            x_mb = x_b[n_minibatch * minibatch_size : , : , : , :]
            y_h_mb = y_b_h[n_minibatch * minibatch_size : , : ]
            if flag:
               loss += net.train_on_batch(x_mb, y_h_mb)
            else:
                loss += net.test_on_batch(x_mb, y_h_mb)
            
            loss= loss/(n_minibatch + 1)
        else:
            loss= loss/n_minibatch

        return loss
   
#%% 
def train_test( dataset, batch_size, flag):
     # Loading Images and training the network per batch
     n_batch = len(dataset) // batch_size
     loss = np.zeros((1 , 2))

     for  batch in range(n_batch):
        dataset_b = dataset[batch * batch_size : (batch + 1) * batch_size ]
        try:
            loss += rut(dataset_b, flag, patches_size)
        except FileExistsError:
            print('Image not found')
            continue

    # Getting the remaining batch of the training set  
     if len(dataset) % batch_size:        
        dataset_b = dataset[n_batch * batch_size : ] 
        loss += rut(dataset_b, flag, patches_size)

        loss= loss/(n_batch + 1)
     else:
        loss = loss/n_batch
                    
     return loss
#%%    
def Train(batch_size, epochs, test_set, train_set): 
    
    global net, k
             
#    This is if I use validation set 
    train_set = train_set[round(0.1*len(train_set)):]
    validation_set = train_set[ : round(0.1*len(train_set))]

    loss_train_plot = []
    accuracy_train = []   

    loss_val_plot = []
    accuracy_val = []

    loss_test_plot = []
    accuracy_test = []
   
    patience_cnt = 0
    minimum = 10000.0
    
    print('Start the training')
    start = time.time()
    
    for epoch in range(epochs):
        
        loss_train = np.zeros((1 , 2))
        loss_test = np.zeros((1 , 2))

        # Shuffling the train data 
        train_set = shuffle(train_set, random_state = 0)
        
      # Evaluating the network in the train set
        loss_train = train_test(train_set, batch_size, flag = 1)
        
        print("%d [training loss: %f, Train Acc: %.2f%%]" %(epoch, loss_train[0,0], 100*loss_train[0,1]))
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        
        loss_train_plot.append(loss_train[0,0])
        accuracy_train.append(100*loss_train[0,1])        
      
        if not ((epoch) % 1):
            if Arq ==1:
                net.save('Segnet_%d_%d.h5'%(epoch,k))
            elif Arq==2:
                net.save('Unet_%d_%d.h5'%(epoch,k))
            elif Arq==3:
                net.save('Deep_%d_%d.h5'%(epoch,k))
            else :
               net.save('Dense_%d_%d.h5'%(epoch,k))

         ################################################
        # Evaluating the network in the validation set
        
        loss_val = train_test(validation_set, batch_size, flag = 0)
        
        loss_val_plot.append(loss_val[0,0])
        accuracy_val.append(100*loss_val[0,1])        
        print(k)
#        Performing Early stopping
        if  loss_val[0,0] < minimum:
          patience_cnt = 0
          minimum = loss_val[0,0]
          if Arq ==1:
              net.save('best_model_Segnet_' + str(k) + '.h5')
          elif Arq== 2 :
              net.save('best_model_Unet_' + str(k) + '.h5')
          elif Arq== 3 :
              net.save('best_model_Deep_' + str(k) + '.h5')
          else:
              net.save('best_model_Dense_' + str(k) + '.h5')
              
        else:
          patience_cnt += 1
#
        print("%d [Validation loss: %f, Validation Acc: %.2f%%]" %(epoch , loss_val[0 , 0], 100 * loss_val[0 , 1]))
    
        if patience_cnt > 10:
          print("early stopping...")
          break
    
    #del net


    #if Arq ==1:
        #net = Segnet(nClasses = 2, optimizer = None, input_width = 512 , input_height = 512 , nChannels = 3)
        #net.load_weights('best_model_Segnet_' + str(k) + '.h5')
    #elif Arq ==2:
        #net = Unet(2, None, patches_size, patches_size , 3) 
        #net.load_weights('best_model_Unet_' + str(k) + '.h5')
    #elif Arq ==3:
        #net = Deeplabv3p(weights=None, input_tensor=None, infer = False,
                    #input_shape=(512, 512, 3), classes=2, backbone='xception', OS=16, alpha=1.)
        #net.load_weights('best_model_Deep_' + str(k) + '.h5')
    #else :
        #net = Tiramisu(input_shape=(512,512,3), n_classes = 2,n_filters_first_conv = 32, n_pool = 8, growth_rate = 8 , 
                        #n_layers_per_block = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],  dropout_p = 0)
        #net.load_weights('best_model_Dense_' + str(k) + '.h5')  
        
       ################################################
    ##Evaluating the network in the test set

    #loss_test = train_test(test_set, batch_size, flag = 0)
    
    #loss_test_plot.append(loss_test[0,0])
    #accuracy_test.append(100*loss_test[0,1])        

    #print("[Test loss: %f , Test Acc: %.2f%%]" %(loss_test[0,0], 100*loss_test[0,1]))
    
    return loss_train_plot, accuracy_train, loss_val_plot, accuracy_val, loss_test_plot, accuracy_test, 
    
#%%

if __name__=='__main__':
    
    global net, k, Arq
    from Arquitecturas.U_net import Unet
    #from Arquitecturas.Segnet import Segnet
    from Arquitecturas.segnet_unpooling import Segnet
    from Arquitecturas.deeplabv3p import Deeplabv3p
    from Arquitecturas.DenseNet import Tiramisu
       
    Images = Directory()
    n_folds = 5
    overlap_percent = 0
    Arq = 2

    size = len(Images)//n_folds
    
    for k in range(n_folds):       

        test_set = Images[size * k: size * (k+1)]
        train_set = Images[: size * k] + Images[size * (k + 1) :]

        print(len(train_set))
        print(len(test_set))
        
        f = open('file_name_%d.txt'%(k),'w')
        for i in test_set:
            f.write(i + '\n')
        f.close()
        
        if Arq == 1:
            print("Start Segnet"+ str(k))
            patches_size = 512
            net = Segnet(nClasses = 2, optimizer = None, input_width = patches_size , input_height = patches_size , nChannels = 3)
            net.summary()
            opt = Adam(lr=0.0001)
            net.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=["accuracy"])
            #Calling the train function  
            loss_train_Segnet, acc_train_Segnet, loss_val_Segnet, acc_val_Segnet,loss_test_Segnet, acc_test_Segnet= Train(4, 100,test_set,train_set)
            print("Finish Segnet"+ str(k))
        
        elif Arq == 2:
            print("Start Unet"+ str(k))
            patches_size = 512
            net = Unet(2, patches_size, patches_size , 3) 
            net.summary()
            opt = Adam(lr=0.0001)
            net.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=["accuracy"])
            # Calling the train function  
            loss_train_Unet, acc_train_Unet, loss_val_Unet, acc_val_Unet, loss_test_Unet, acc_test_Unet = Train(4,100,test_set,train_set)
            print("Finish Unet"+ str(k))
            k = k+1
        
        elif Arq ==3:
            print("Start DeepLabv3p")
            patches_size = 512
            net = Deeplabv3p(input_tensor=None, infer = False,
                    input_shape=(patches_size, patches_size, 3), classes=2, backbone='mobilenetv2', OS=16, alpha=1.)
            net.summary()
            opt = Adam(lr=0.0001)
            net.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=["accuracy"])
            ## Calling the train function
            loss_train_Deep, acc_train_Deep, loss_val_Deep, acc_val_Deep, loss_test_Deep, acc_test_Deep = Train(4, 100,test_set,train_set)
            print("Finish Deeplabv3p")
        
        else:           
            print("Start DenseNet")
            patches_size = 512
            #net = Tiramisu(input_shape=(512,512,3), n_classes = 2, n_filters_first_conv = 48, n_pool = 5, growth_rate = 16 , n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4], dropout_p = 0.2)
            net = Tiramisu(input_shape=(patches_size,patches_size,3), n_classes = 2,n_filters_first_conv = 32, n_pool = 8, growth_rate = 8 , 
                        n_layers_per_block = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],  dropout_p = 0)
            opt = Adam(lr=0.0001)
            net.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=["accuracy"])
            # Calling the train function  
            loss_train_Deep, acc_train_Deep, loss_val_Deep, acc_val_Deep, loss_test_Deep, acc_test_Deep = Train(4, 100,test_set,train_set)
            print("Finish DenseNet")
  

              
     

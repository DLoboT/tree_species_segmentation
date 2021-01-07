#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:36:53 2019

@author: daliana
"""
#%%

from keras.models import Model, load_model, save_model
from keras import applications
from keras import backend as K
import keras
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import glob
from random import randint
import h5py
from project_patches import Directory, load_im, patches
import cv2
from sklearn.metrics import confusion_matrix,precision_score, accuracy_score, recall_score, f1_score
import time
#from prueba import *
from keras.optimizers import SGD, Adam
from CRF import do_crf
from sklearn import preprocessing as pp

#%% 
def compute_metrics(true_labels, predicted_labels):
       
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # precision tp / (tp + fp)
    precision = 100* precision_score(true_labels, predicted_labels, average='binary')
    
    # recall: tp / (tp + fn)
    recall = 100*recall_score(true_labels, predicted_labels, average='binary')
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1score = 100*f1_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, f1score, recall, precision

def iou_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(np.abs(y_true * y_pred))
    union = np.sum(y_true) + np.sum(y_pred)-intersection
    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou

#%%
def add_padding(im, flag, overlap, stride):
    
    h, w= im.shape[1], im.shape[2]

    step_row = (stride - h % stride) % stride
    step_col = (stride - w % stride) % stride
    
    if flag == 0:
       npad_img = ( (0,0), (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col) )
    else:
       npad_img = ( (0,0), (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col), (0, 0) )
   
    pad_img = np.pad(im, npad_img, mode='symmetric')
    
    return pad_img

#%%            
# Reconstruction
def unblockshaped(predict,pad_img, k1, k2):
 
    _, row, col, _ = pad_img.shape

    nchannels = predict.shape[3]

    t = 0
    reconstructed = np.zeros((row, col, nchannels))
    
    for i in range(k1):
        for j in range(k2):
            reconstructed[i*stride : i*stride+stride, j*stride : j*stride+stride,:] = predict[t, overlap//2 : overlap//2 + stride, 
                                                                                                overlap//2 : overlap//2 + stride,:]
            t = t+1

    return reconstructed
#%%  
def gray2rgb(image):
    """
    Funtion to convert classes values from 0,1,3,4 to rgb values
    """
    row,col = image.shape
    image = image.reshape((row*col))
    rgb_output = np.zeros((row*col, 3))
    rgb_map = [[0,0,255],[0,255,0],[0,255,255],[255,255,0],[255,255,255]]
    for j in np.unique(image):
        rgb_output[image==j] = np.array(rgb_map[j])
    
    rgb_output = rgb_output.reshape((row,col,3))  
    rgb_output = cv2.cvtColor(rgb_output.astype('uint8'),cv2.COLOR_BGR2RGB)
    return rgb_output 


#%%  
def test(net, dataset, k, CRF = True):

    # Loading Images and training the network per batch
    try:
        images, groundtruth, files_name = load_im(dataset, patches_size)
    except FileExistsError:
        print('Image not found')
        raise FileExistsError

    q, h, w, _ = images.shape
    batch_size = 10   #This is for one image
    n_batch = len(images)//batch_size
    global idb
    idb = 0
   
    def Routine_on_Batch(image_pad, groundtruth_pad):
        global idb, CRF

        patches_array, patches_ref, k1, k2 = patches(image_pad, groundtruth_pad, patches_size, overlap_percent = P)
        
        if Arq !=3:
            patches_array = patches_array.astype('float32')/255
        
        predict_probs = net.predict(patches_array, verbose=1,batch_size = 16)
        
        predict_labels = predict_probs.argmax(axis=-1)
        
        size = k1 * k2
        for i in range(image_pad.shape[0]):
            
            im_ref = unblockshaped(np.expand_dims(patches_ref[i*size:(i+1)*size], axis=3),image_pad, k1, k2)
            im_ref = np.squeeze(im_ref)
            im_ref = im_ref[:h,:w].astype('uint8')
         
            if CRF:
                #    Applying CRF   
                reconstructed_probs = unblockshaped(predict_probs[i*size:(i+1)*size],image_pad, k1, k2)
                reconstructed_probs = reconstructed_probs[:h,:w, :]
                start = time.time()
                segmented_img = do_crf(images[i], reconstructed_probs)
                end = time.time()
                print("CRF time: %2f" %(end - start))
            else:
                segmented_img = unblockshaped(np.expand_dims(predict_labels[i*size:(i+1)*size], axis=3),image_pad, k1, k2)
                segmented_img = np.squeeze(segmented_img)
                segmented_img = segmented_img[:h,:w].astype('uint8')
         
            segmented_img = gray2rgb(segmented_img)
            im_ref = gray2rgb(im_ref)
            
            print('segmented_img')
            print(segmented_img.shape)

            file_name = filename[idb]  + ".tiff"
            cv2.imwrite(os.path.join(segmented_img_dir , file_name) ,segmented_img)
           
            file_name = filename[idb] + "_gdt" + ".tiff"
            cv2.imwrite(os.path.join(reconst_GDT_dir , file_name), im_ref)
            idb += 1

    f = open('file_name.txt','w')
    for i in files_name:
        f.write(i + '\n')
    f.close()
#    Storing current images name
    global filename   
    filename = []
    for i in range(q):
        im = files_name[i]
        filenames = im[-23:-13] + '_' + im[-12:-4]

        filename.append(filenames)
    
    for batch in range(n_batch):
        image_pad = add_padding(images[batch * batch_size : (batch + 1) * batch_size ], 1, overlap, stride)
        groundtruth_pad = add_padding(groundtruth[batch * batch_size : (batch + 1) * batch_size ], 0, overlap, stride)
        Routine_on_Batch(image_pad, groundtruth_pad)

#    Last batch!
    if len(images) % batch_size:
        image_pad = add_padding(images[n_batch * batch_size : ] , 1, overlap, stride)
        groundtruth_pad = add_padding(groundtruth[n_batch * batch_size :], 0, overlap, stride)
        Routine_on_Batch(image_pad, groundtruth_pad)
    

#%%  
if __name__=='__main__':
 
    from Arquitecturas.U_net import Unet
    from Arquitecturas.segnet_unpooling import Segnet
    from Arquitecturas.deeplabv3p import Deeplabv3p
#    from Arquitecturas.DenseNet import Tiramisu
    
    global CRF, Arq
   
    CRF = False
    Arq = 3
    patches_size = 512
    P = 0
    overlap = round(patches_size * P)
    overlap -= overlap % 2
    stride = patches_size - overlap
#    stride = patches_size//2

#    train_set, test_set = Directory()

    for k in range(5):

#        Creating Directories
        if CRF:
            segmented_img_dir = 'Resultados/Predict_CRF/' + str(k) + '/'
        else:
            segmented_img_dir = 'Resultados/Predict/' + str(k) + '/'
        if not os.path.isdir(segmented_img_dir):
            os.mkdir(segmented_img_dir)
        
        reconst_GDT_dir = './Resultados/GDT/' + str(k) + '/'
        if not os.path.isdir(reconst_GDT_dir):
            os.mkdir(reconst_GDT_dir)

        
        f1 = open("file_name_%d.txt"%(k), "r")
        test_set = f1.readlines()
        f1.close()

        for j in range(len(test_set)):
            test_set[j] = test_set[j][:-1]
        if Arq == 1:
            # To load the weigths Segnet    
            net = Segnet(nClasses = 2, optimizer = None, input_width = patches_size , input_height = patches_size , nChannels = 3)
            net.load_weights('best_model_Segnet_%d.h5'%(k))  
    
        elif Arq == 2:
    
            # To load the weigths Unet    
            net = Unet(2, patches_size, patches_size , 3) 
            net.load_weights('best_model_Unet_%d.h5'%(k))
        
        elif Arq == 3:
    
            # To load the weigths Deeplabv3p    
            net = Deeplabv3p(weights=None, input_tensor=None, infer = False,
                 input_shape=(512, 512, 3), classes=2, backbone='mobilenetv2', OS=16, alpha=1.)
            net.load_weights('best_model_Deep_%d.h5'%(k))
        
        else:
            net = Tiramisu(input_shape=(512,512,3), n_classes = 2, n_filters_first_conv = 32, n_pool = 8, growth_rate = 8, n_layers_per_block = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dropout_p = 0)
            net.load_weights('best_model_Dense_%d.h5'%(k))

        net.summary()

        test(net, test_set, k, CRF)
        
        print('%d-fold ended' %(k))

        # get metrics
        if CRF:
            segmented_imgs = os.listdir(segmented_img_dir)
        else:
            segmented_imgs = os.listdir(segmented_img_dir)
           
        intersection = 0
        union = 0
        tnegative = 0
        fpositive = 0
        fnegative = 0
        tpositive = 0
        
        for j in range(len(segmented_imgs)): 
            pred = Image.open(segmented_img_dir + filename[j] + ".tiff")
            GDTrue = Image.open(reconst_GDT_dir + filename[j] + "_gdt" + ".tiff")
            
#            The class cumbarú is segmented on the 2nd (green) channel
            pred = np.asarray(pred)[:,:,1] // 255
            GDTrue= np.asarray(GDTrue)[:,:,1] // 255
            
            #pred_mask = pred.flatten()
            #pred_mask = keras.utils.to_categorical(pred_mask,2)
            #pred_mask = pred_mask.reshape(pred.shape[0], pred.shape[1], -1) 

            #GDTrue_mask = GDTrue.flatten()
            #GDTrue_mask = keras.utils.to_categorical(GDTrue_mask,2)
            #GDTrue_mask = GDTrue_mask.reshape(GDTrue.shape[0],GDTrue.shape[1],-1) 
                
            #iou = iou_coef(GDTrue_mask, pred_mask, smooth=1)
            #Iou += iou

            img_reconstructed = pred.flatten()
            img_ref = GDTrue.flatten()

#            acc, f1s, rec, prec = compute_metrics(img_ref, img_reconstructed)

            tn, fp, fn, tp = confusion_matrix(img_ref, img_reconstructed).ravel()
#            print(tn, fp, fn, tp)
            
            intersection += np.sum(np.logical_and(pred, GDTrue)) # Logical AND  
            union += np.sum(np.logical_or(pred, GDTrue)) # Logical OR 
            
            tnegative += tn
            fpositive += fp
            fnegative += fn
            tpositive += tp
            
#            accuracy += acc
#            f1score += f1s
#            recall += rec
#            precision += prec
#            
            
#        accuracy /= len(segmented_imgs)
#        f1score /= len(segmented_imgs)
#        recall /= len(segmented_imgs)
#        precision /= len(segmented_imgs)
              
        Iou = intersection/union
        accu = (tpositive + tnegative)/(tnegative + fpositive + fnegative + tpositive)
        Prec = tpositive/(tpositive + fpositive)
        R = tpositive/(tpositive + fnegative)
        F1 = 2*Prec*R/(Prec+R)

        
#        tnegative /= len(segmented_imgs)
#        fpositive /= len(segmented_imgs)
#        fnegative /= len(segmented_imgs)
#        tpositive /= len(segmented_imgs)
        
        #Iou /= len(segmented_imgs)

         
        print('Test accuracy:%.2f' %(100*accu))
        print('Test f1score:%.2f' %(100*F1))
        print('Test prescision:%.2f' %(100*Prec))
        print('Test recall:%.2f' %(100*R))
        print('Intersection over Union:%.2f' %(100*Iou))
        print('Confusion_matrix')
        print('True negative:%.2f' %(tnegative))
        print('False positive:%.2f' %(fpositive))
        print('False negative:%.2f' %(fnegative))
        print('True positive:%.2f' %(tpositive))

        lt = 'a' 
        if not k:
            lt = 'w'
        if Arq == 1:
            file_metrics = open("metrics_Segnet_%d.txt"%(P), lt)
        elif Arq == 2:
            file_metrics = open("metrics_Unet_%d.txt"%(P), lt)
        elif Arq == 3:
            file_metrics = open("metrics_DeepLab_%d.txt"%(P), lt)
        else:
            file_metrics = open("metrics_FCDenseNet_%d.txt"%(P), lt)
        
        file_metrics.write('K-Fold:%d\n'%(k))
        file_metrics.write('Acc:%2f\n'%(100*accu))
        file_metrics.write('F1:%2f\n'%(100*F1))
        file_metrics.write('Recall:%2f\n'%(100*R))
        file_metrics.write('Precision:%2f\n'%(100*Prec))
        file_metrics.write('IoU:%2f\n\n'%(100*Iou))
        
        file_metrics.write('Confusion_matrix\n\n')
        file_metrics.write('TN:%2f\n\n'%(tnegative))
        file_metrics.write('FP:%2f\n\n'%(fpositive))
        file_metrics.write('FN:%2f\n\n'%(fnegative))
        file_metrics.write('TP:%2f\n\n'%(tpositive))
        
        file_metrics.close()

       
            

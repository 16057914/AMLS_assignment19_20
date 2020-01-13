#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#this contains all preprocessing required for B tasks, both with (for B1) and without feature selection (for B2).

from sklearn.model_selection import train_test_split
import os
from keras.preprocessing import image
import numpy as np
from featureext import run_dlib_shape

def extract_features_labels_from_cartoon(x): #extracts features, images and labels from the celebritiy images folder.
    
    # Global Parameters
    basedir = './Datasets'
    images_dir = os.path.join(basedir,'cartoonimg')
    labels_filename = 'cartoonlabels.csv'

    # Setting paths of images and labels
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    if x == 1:
        target_size = None #leave them for feature extraction in B1
    elif x == 2:   
        target_size = (64,64) #resize images for B2
    
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    
    # Obtaining the labels
    lines = labels_file.readlines()
    lines = [line.strip('"\n') for line in lines[:]]
    eye_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    face_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    
    # Extract landmark features and labels
    if os.path.isdir(images_dir):
        all_images = []
        all_eye_labels = []

        all_features = []
        fe_face_labels = []
        for img_path in image_paths:
            if not img_path.endswith('.png'):
                continue
            file_name= img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            all_images.append(img) #append the images to save all images, for testing without feature extraction
            all_eye_labels.append(eye_labels[file_name]) #append all eye colour labels, only B2 needs pure images.
            
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features) #append features if obtained
                fe_face_labels.append(face_labels[file_name]) #and if so, append face shape labels, only B1 needs feature extraction.
                
    celebimages = np.array(all_images)
    eye_labels = np.array(all_eye_labels)
    
    landmark_features = np.array(all_features)
    fe_face_labels = np.array(fe_face_labels)
    
    #returns differ for tasks B1 and B2.
    if x == 1:
        return landmark_features,fe_face_labels
    elif x == 2:
        return celebimages, eye_labels

def extract_features_labels_from_cartoon_test(x): #same as above, for the test dataset

    # Global Parameters
    basedir = './Datasets'
    images_dir = os.path.join(basedir,'cartoonimg_test')
    labels_filename = 'cartoonlabels_test.csv'

    # Setting paths of images and labels
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    if x == 1:
        target_size = None #leave them for feature extraction in B1
    elif x == 2:   
        target_size = (64,64) #resize images for B2

    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    
    # Obtaining the labels
    lines = labels_file.readlines()
    lines = [line.strip('"\n') for line in lines[:]]
    eye_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    face_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    
    # Extract landmark features and labels
    if os.path.isdir(images_dir):
        all_images = []
        all_eye_labels = []

        all_features = []
        fe_face_labels = []
        for img_path in image_paths:
            if not img_path.endswith('.png'):
                continue
            file_name= img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            all_images.append(img)
            all_eye_labels.append(eye_labels[file_name])
            
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                fe_face_labels.append(face_labels[file_name])
                
    celebimages = np.array(all_images)
    eye_labels = np.array(all_eye_labels)
    
    landmark_features = np.array(all_features)
    fe_face_labels = np.array(fe_face_labels)
    
    if x == 1:
        return landmark_features,fe_face_labels
    elif x == 2:
        return celebimages, eye_labels

def preprocess_B1(): #preprocessing required for B1
    
    landmarkfeatures,facelabels = extract_features_labels_from_cartoon(1)
    testlandmarkfeatures,testfacelabels = extract_features_labels_from_cartoon_test(1)
    
    X = landmarkfeatures
    y = facelabels
    
    d2_X = X.reshape(len(X), 68*2)
    
    X_test = testlandmarkfeatures
    y_test = testfacelabels
    
    d2_X_test = X_test.reshape(len(X_test), 68*2)
    
    X_train, X_val, y_train, y_val = train_test_split(d2_X, y, test_size=0.2, random_state=42)
    
    return d2_X, y, X_train, X_val, d2_X_test, y_train, y_val, y_test 

def preprocess_B2(x): #preprocessing for B2 
    #x = 0 for svm, x = 1 for cnn. CNN requires different inputs (non flattened X, one hot encoded y)
    cartoonimages,eyelabels = extract_features_labels_from_cartoon(2)
    testcartoonimages,testeyelabels = extract_features_labels_from_cartoon_test(2)
    
    X = cartoonimages
    y = eyelabels
    
    d2_X = X.reshape((10000,64*64*3)) #for svm
    
    one_hot_y = np.zeros((10000,5)) #for cnn
    for i in range(10000):
        one_hot_y[i, y[i]] = 1
    
    X_test = testcartoonimages
    y_test = testeyelabels
    
    ntestsamples, ntestx, ntesty, ntestz = X_test.shape #for svm
    d2_X_test = X_test.reshape((ntestsamples,ntestx*ntesty*ntestz))
    
    onehot_ytest = np.zeros((2500,5)) #for cnn
    for i in range(2500):
        onehot_ytest[i, y_test[i]] = 1

    
    d2_X_train, d2_X_val, y_train, y_val = train_test_split(d2_X, y, test_size=0.2, random_state=42)
    
    X_train, X_val, onehot_y_train, onehot_y_val = train_test_split(X, one_hot_y, test_size=0.2, random_state=42)
        
    if x == 0:
        return d2_X, y, d2_X_train, d2_X_val, d2_X_test, y_train, y_val, y_test #return d2_X for learning curve if want
    
    if x == 1:
        return X_train, X_val, X_test, onehot_y_train, onehot_y_val, onehot_ytest
    
    
    
    

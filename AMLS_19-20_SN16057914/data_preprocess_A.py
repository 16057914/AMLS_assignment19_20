#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing import image
import numpy as np
from featureext import run_dlib_shape
from skimage import color

def extract_features_labels_from_celeb(x):
    
    # Global Parameters
    basedir = './Datasets'
    images_dir = os.path.join(basedir,'celebimg')
    labels_filename = 'celeblabels.csv'

    # Setting paths of images and labels
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    
    # Obtaining the labels
    lines = labels_file.readlines()
    lines = [line.strip('"\n') for line in lines[:]]
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    smiling_labels = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    
    # Extract landmark features and labels
    if os.path.isdir(images_dir):
        all_images = []
        all_gender_labels = []
        all_smiling_labels = []
        all_features = []
        fe_gender_labels = []
        fe_smiling_labels = []
        for img_path in image_paths:
            if not img_path.endswith('.jpg'):
                continue
            file_name= img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            all_images.append(img)
            all_gender_labels.append(gender_labels[file_name])
            all_smiling_labels.append(smiling_labels[file_name])
            
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                fe_gender_labels.append(gender_labels[file_name])
                fe_smiling_labels.append(smiling_labels[file_name])
                
    celebimages = np.array(all_images)
    gender_labels = np.array(all_gender_labels)
    smiling_labels = np.array(all_smiling_labels)
    
    landmark_features = np.array(all_features)
    fe_gender_labels = np.array(fe_gender_labels)
    fe_smiling_labels = np.array(fe_smiling_labels)
    
    if x == 1:
        return landmark_features, fe_gender_labels
    elif x == 11:
        return celebimages, gender_labels
    elif x == 2:
        return landmark_features, fe_smiling_labels
    elif x == 22:
        return celebimages, smiling_labels

def extract_features_labels_from_celeb_test(x):

    # Global Parameters
    basedir = './Datasets'
    images_dir = os.path.join(basedir,'celebimg_test')
    labels_filename = 'celeblabels_test.csv'

    # Setting paths of images and labels
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    
    # Obtaining the labels
    lines = labels_file.readlines()
    lines = [line.strip('"\n') for line in lines[:]]
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    smiling_labels = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    
    # Extract landmark features and labels
    if os.path.isdir(images_dir):
        all_images = []
        all_gender_labels = []
        all_smiling_labels = []
        all_features = []
        fe_gender_labels = []
        fe_smiling_labels = []
        for img_path in image_paths:
            if not img_path.endswith('.jpg'):
                continue
            file_name= img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            all_images.append(img)
            all_gender_labels.append(gender_labels[file_name])
            all_smiling_labels.append(smiling_labels[file_name])
            
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                fe_gender_labels.append(gender_labels[file_name])
                fe_smiling_labels.append(smiling_labels[file_name])
                
    celebimages = np.array(all_images)
    gender_labels = np.array(all_gender_labels)
    smiling_labels = np.array(all_smiling_labels)
    
    landmark_features = np.array(all_features)
    fe_gender_labels = np.array(fe_gender_labels)
    fe_smiling_labels = np.array(fe_smiling_labels)
    
    if x == 1:
        return landmark_features, fe_gender_labels
    elif x == 11:
        return celebimages, gender_labels
    elif x == 2:
        return landmark_features, fe_smiling_labels
    elif x == 22:
        return celebimages, smiling_labels
    
def preprocess_A1():
    
    landmarkfeatures, genderlabels = extract_features_labels_from_celeb(1)
    testlandmarkfeatures, testgender = extract_features_labels_from_celeb_test(1)
    
    X = landmarkfeatures
    y = genderlabels
    
    d2_X = X.reshape(len(X), 68*2)
    
    X_test = testlandmarkfeatures
    y_test = testgender
    
    d2_X_test = X_test.reshape(len(X_test), 68*2)
    
    X_train, X_val, y_train, y_val = train_test_split(d2_X, y, test_size=0.2, random_state=42)
    
    return d2_X,y,X_train, X_val, d2_X_test, y_train, y_val, y_test

def preprocess_A2():
    
    landmarkfeatures, smilinglabels = extract_features_labels_from_celeb(2)
    testlandmarkfeatures, testsmiling = extract_features_labels_from_celeb_test(2)
    
    X = landmarkfeatures
    y = smilinglabels
    
    d2_X = X.reshape(len(X), 68*2)
    
    X_test = testlandmarkfeatures
    y_test = testsmiling
    
    d2_X_test = X_test.reshape(len(X_test), 68*2)
    
    X_train, X_val, y_train, y_val = train_test_split(d2_X, y, test_size=0.2, random_state=42)
    
    return d2_X,y,X_train, X_val, d2_X_test, y_train, y_val, y_test
    
def preprocess_images_A1():
    
    celebimages, allgenderlabels = extract_features_labels_from_celeb(11)
    testcelebimages, testallgender = extract_features_labels_from_celeb_test(11)
    
    #convert to greyscale
    greyimages = []
    for i in celebimages:
        img = color.rgb2gray(i)
        greyimages.append(img)
        
    X = np.asarray(greyimages)
    y = allgenderlabels
    
    nsamples, nx, ny = X.shape
    d2_X = X.reshape((nsamples,nx*ny))
    
    X_train, X_val, y_train, y_val = train_test_split(d2_X, y, test_size=0.2, random_state=42)
    
    greytestimages = []
    for l in testcelebimages:
        testimg = color.rgb2gray(l)
        greytestimages.append(testimg)
    
    X_test = np.asarray(greytestimages)
    y_test = testallgender
    
    ntestsamples, ntestx, ntesty = X_test.shape
    d2_X_test = X_test.reshape((ntestsamples,ntestx*ntesty))
    
    return d2_X,y,X_train, X_val, d2_X_test, y_train, y_val, y_test
    
def preprocess_images_A2():
    
    celebimages, allsmilinglabels = extract_features_labels_from_celeb(22)
    testcelebimages, testallsmiling = extract_features_labels_from_celeb_test(22)
    
    #convert to greyscale
    greyimages = []
    for i in celebimages:
        img = color.rgb2gray(i)
        greyimages.append(img)
        
    X = np.asarray(greyimages)
    y = allsmilinglabels
    
    nsamples, nx, ny = X.shape
    d2_X = X.reshape((nsamples,nx*ny))
    
    X_train, X_val, y_train, y_val = train_test_split(d2_X, y, test_size=0.2, random_state=42)
    
    greytestimages = []
    for l in testcelebimages:
        testimg = color.rgb2gray(l)
        greytestimages.append(testimg)
    
    X_test = np.asarray(greytestimages)
    y_test = testallsmiling
    
    ntestsamples, ntestx, ntesty = X_test.shape
    d2_X_test = X_test.reshape((ntestsamples,ntestx*ntesty))
    
    return d2_X,y, X_train, X_val, d2_X_test, y_train, y_val, y_test

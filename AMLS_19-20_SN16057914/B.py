#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam, Adadelta
from keras.losses import categorical_crossentropy

def svmtrain(X_train,y_train,kernel_type):
    svm_model = SVC(kernel = kernel_type, C = 1)
    
    svm_model.fit(X_train, y_train)
    
    return svm_model
    
def svmval(X_val,y_val,svm_model,X_train,y_train):
    
    predictions = svm_model.predict(X_val)
    
    y_pred_train = svm_model.predict(X_train)
    
    valacc = accuracy_score(y_val,predictions)
    
    trainacc = accuracy_score(y_train,y_pred_train)
    
    return trainacc, valacc, predictions

def svmtest(X_test,y_test,svm_model):
    
    testpred = svm_model.predict(X_test)
    
    testacc = accuracy_score(y_test,testpred)
    
    return testacc


def cnntrain(X_train,y_train,X_test,y_test,optimiser):
    model_cnn = Sequential()
    # specify input shape
    model_cnn.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(64,64,3))) #image size
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(5, activation='softmax')) #5 classes
    
    #compile
    model_cnn.compile(loss='categorical_crossentropy',
              optimizer=optimiser,
              metrics=['accuracy'])
    
    model_cnn.fit(X_train,y_train,
          batch_size=60,
          epochs=3,
          verbose=1,
          validation_data=(X_test,y_test))
    
def svm_lc(X,y,x):
    
    if x == 1: 
        trainsizes = [2000,3000,5000,6552]
        
    elif x == 2: 
        trainsizes = [2000, 4000, 6000, 7997]
    
    train_sizes, train_scores, valid_scores = learning_curve(SVC(C=1,kernel='linear'), X, y, train_sizes=trainsizes, cv=5)
    
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = valid_scores.mean(axis = 1)
    
    
    return train_sizes, train_scores_mean,validation_scores_mean
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

#from sklearn.svm import SVC

def logregtrain(X_train,y_train):
    logreg = LogisticRegression()
    
    logreg.fit(X_train,y_train)
    
    return logreg

def logregval(X_val,y_val,logregmodel,X_train,y_train):
    
    predictions = logregmodel.predict(X_val)
    
    y_pred_train = logregmodel.predict(X_train)
    
    valacc = accuracy_score(y_val,predictions)
    
    trainacc = accuracy_score(y_train,y_pred_train)
    
    return trainacc, valacc, predictions #so can do confmatrix after
    
def logregtest(X_test,y_test,logregmodel):
    
    testpred = logregmodel.predict(X_test)
    
    testacc = accuracy_score(y_test,testpred)
    
    return testacc
    
    
def logreg_lc(X,y):
    train_sizes, train_scores, valid_scores = learning_curve(LogisticRegression(), X, y, train_sizes=[500,1000, 2000,3000, 3835], cv=5)
    
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = valid_scores.mean(axis = 1)
    
    
    return train_sizes, train_scores_mean,validation_scores_mean
    
#def SVM(X_train,y_train,X_val,y_val):
 #   svmclf = SVC(kernel='linear',C=1)
    

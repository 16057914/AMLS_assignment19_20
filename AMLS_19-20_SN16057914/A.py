#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This class A.py contains the functions required to build, validate and test the model used in tasks A.
#Also contains function to create a learning curve.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve


def logregtrain(X_train,y_train): #this builds the logistic regression and fits it to the train data
    logreg = LogisticRegression()
    
    logreg.fit(X_train,y_train)
    
    return logreg

def logregval(X_val,y_val,logregmodel,X_train,y_train): #this validates the model passed to it
    
    predictions = logregmodel.predict(X_val)
    
    y_pred_train = logregmodel.predict(X_train)
    
    valacc = accuracy_score(y_val,predictions)
    
    trainacc = accuracy_score(y_train,y_pred_train)
    
    return trainacc, valacc, predictions #return predictions so the confusion matrix may be plotted
    
def logregtest(X_test,y_test,logregmodel): #this tests the model using the separate test data
    
    testpred = logregmodel.predict(X_test)
    
    testacc = accuracy_score(y_test,testpred)
    
    return testacc
    
    
def logreg_lc(X,y): #this enables the learning curve to be plotted
    #it finds the cross validated training and validation accuracies for different training set sizes.
    train_sizes, train_scores, valid_scores = learning_curve(LogisticRegression(), X, y, train_sizes=[500,1000, 2000,3000, 3835], cv=5)
    
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = valid_scores.mean(axis = 1)
    
    
    return train_sizes, train_scores_mean,validation_scores_mean
    

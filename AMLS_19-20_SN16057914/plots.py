#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#the functions to plot graphs discussed.

from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

def plotconfmat(accscore,predictions,y_test,name): #confusion matrix
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(accscore)
    plt.title(all_sample_title, size = 15);
    plt.savefig(name+"confmat.png")

def plot_lc(train_sizes, train_scores_mean,validation_scores_mean,name): #learning curve

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for Task '+name, fontsize = 16, y = 1.03)
    plt.legend()
    plt.ylim(0.7,1.1)
    plt.savefig(name+"lc.png")

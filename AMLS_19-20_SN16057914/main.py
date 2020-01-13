from A import logregtrain, logregval, logregtest, logreg_lc
from B import svmtrain, svmval, svmtest,svm_lc,cnntrain
from data_preprocess_A import preprocess_A1,preprocess_A2, preprocess_images_A1, preprocess_images_A2
from data_preprocess_B import preprocess_B1, preprocess_B2
from plots import plotconfmat, plot_lc

#data preprocessing
#task A1 with feature selection
d2_XA1, y_A1, X_trainA1, X_valA1, X_testA1, y_trainA1, y_valA1, y_testA1 = preprocess_A1()

#task A2 with feature selection
d2_XA2, y_A2, X_trainA2, X_valA2, X_testA2, y_trainA2, y_valA2, y_testA2 = preprocess_A2()


#task B1
d2_XB1, y_B1, X_trainB1, X_valB1, X_testB1, y_trainB1, y_valB1, y_testB1 = preprocess_B1()

#task B2
d2_XB2, y_B2, X_trainB2, X_valB2, X_testB2, y_trainB2, y_valB2, y_testB2 = preprocess_B2(0)



#==============================================================================
#task A1, logistic regression with feature extraction
logregmodelA1 = logregtrain(X_trainA1, y_trainA1)
acc_A1_train, acc_A1_valid, predictionsA1 = logregval(X_valA1,y_valA1,logregmodelA1,X_trainA1,y_trainA1)

#final test accuracy on selected model
acc_A1_test = logregtest(X_testA1,y_testA1,logregmodelA1)

#if want to do learning curve for A1 and while doing so, obtain cross-validation scores
#saves plot
#A1_train_sizes, A1_train_scores, A1_valid_scores = logreg_lc(d2_XA1, y_A1) #obtain train and valid scores from here
#plot_lc(A1_train_sizes, A1_train_scores, A1_valid_scores,'A1')

#to plot confusion matrix on validation data, saves plot
#plotconfmat(acc_A1_valid, predictionsA1, y_valA1, 'A1')


#task A1, testing logistic regression with images
#d2_XA1im, y_A1im, X_trainA1im, X_valA1im, X_testA1im, y_trainA1im, y_valA1im, y_testA1im = preprocess_images_A1()
#logregmodelA1im = logregtrain(X_trainA1im, y_trainA1im)
#acc_A1im_train,acc_A1im_valid, predictionsA1im = logregval(X_valA1im,y_valA1im,logregmodelA1im,X_trainA1im, y_trainA1im)
#acc_A1im_test = logregtest(X_testA1im,y_testA1im,logregmodelA1im)

#==============================================================================
#task A2, logistic regression with feature extraction
logregmodelA2 = logregtrain(X_trainA2,y_trainA2)
acc_A2_train, acc_A2_valid, predictionsA2 = logregval(X_valA2,y_valA2,logregmodelA2,X_trainA2,y_trainA2)

#final test accuracy on selected model
acc_A2_test = logregtest(X_testA2,y_testA2,logregmodelA2)

#if want to do learning curve for A2 and while doing so, obtain cross-validation scores
#saves plot
#A2_train_sizes, A2_train_scores, A2_valid_scores = logreg_lc(d2_XA2, y_A2)
#plot_lc(A2_train_sizes, A2_train_scores, A2_valid_scores,'A2')

#to plot confusion matrix, saves plot
#plotconfmat(acc_A2_valid, predictionsA2, y_valA2, 'A2')

#task A2, testing logistic regression with images
#d2_XA2im, y_A2im, X_trainA2im, X_valA2im, X_testA2im, y_trainA2im, y_valA2im, y_testA2im = preprocess_images_A2()
#logregmodelA2im = logregtrain(X_trainA2im, y_trainA2im)
#acc_A2im_train, acc_A2im_valid, predictionsA2im = logregval(X_valA2im,y_valA2im,logregmodelA2im,X_trainA2im, y_trainA2im)
#acc_A2im_test = logregtest(X_testA2im,y_testA2im,logregmodelA2im)

#==============================================================================
#task B1, SVM with feature extraction, kernel = linear
svm_modelB1 = svmtrain(X_trainB1,y_trainB1,'linear')
acc_B1_train, acc_B1_valid, predictionsB1 = svmval(X_valB1,y_valB1,svm_modelB1,X_trainB1,y_trainB1)

#final test accuracy on selected model
acc_B1_test = svmtest(X_testB1,y_testB1,svm_modelB1)

#if want to do learning curve and while doing so, obtain cross-validation scores
#saves plot
#B1_train_sizes, B1_train_scores, B1_valid_scores = svm_lc(d2_XB1,y_B1,1) 
#plot_lc(B1_train_sizes, B1_train_scores, B1_valid_scores,'B1')

#to plot confusion matrix, saves plot
#plotconfmat(acc_B1_valid, predictionsB1, y_valB1, 'B1')

#task B1 testing with SVM, kernel = 'rbf'
#svm_modelB1rbf = svmtrain(X_trainB1,y_trainB1,'rbf')
#acc_B1_train_rbf, acc_B1_valid_rbf, predictionsB1_rbf = svmval(X_valB1,y_valB1,svm_modelB1rbf,X_trainB1,y_trainB1)

#==============================================================================
#task B2, SVM, kernel = linear
svm_modelB2 = svmtrain(X_trainB2,y_trainB2,'linear')
acc_B2_train, acc_B2_valid, predictionsB2 = svmval(X_valB2,y_valB2,svm_modelB2,X_trainB2,y_trainB2)

#final test accuracy on selected model
acc_B2_test = svmtest(X_testB2,y_testB2,svm_modelB2)

#if want to do learning curve and while doing so, obtain cross-validation scores
#saves plot as figure
#B2_train_sizes, B2_train_scores, B2_valid_scores = svm_lc(d2_XB2,y_B2,2)
#plot_lc(B2_train_sizes, B2_train_scores, B2_valid_scores,'B2')

#to plot confusion matrix, saves plot
#plotconfmat(acc_B2_valid, predictionsB2, y_valB2, 'B2')

#task B2, testing with SVM, kernel = rbf
#svm_modelB2rbf = svmtrain(X_trainB2,y_trainB2,'rbf')
#acc_B2_train_rbf, acc_B2_valid_rbf, predictionsB2_rbf = svmval(X_valB2,y_valB2,svm_modelB2rbf,X_trainB2,y_trainB2)

#task B2, testing with CNN, optimiser = Adadelta (also tested with rmsprop)
#X_traincnn, X_valcnn, X_testcnn, y_traincnn, y_valcnn, y_testcnn = preprocess_B2(1)
#cnntrain(X_traincnn,y_traincnn,X_valcnn,y_valcnn,'Adadelta') #'Adadelta' or 'rmsprop' etc can be inputted

#==============================================================================
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(
        acc_A1_train, acc_A1_test,
        acc_A2_train, acc_A2_test,
        acc_B1_train, acc_B1_test,
        acc_B2_train, acc_B2_test))


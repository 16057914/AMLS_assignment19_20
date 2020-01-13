# AMLS_assignment19_20

All files required for my code to run are in the file 'AMLS_19-20_SN16057914'. Inside will be a main.py that simply needs to be run - it will use the other python files that I have created in this folder (such as preprocess_B and so on). 

There is also a folder named 'Datasets' that is empty on this Github as the file is too large but shows that this is how I want the Datasets folder placed in 'AMLS_19-20_SN16057914'. I provide a Google Drive link to download the Datasets folder with the data and labels named in the way I refer to them in my code. It's important that this Dataset folder is downloaded and placed in the 'AMLS_19-20_SN16057914' folder for my code to work.

The 'shape_predictor_68_face_landmarks.dat' must also be downloaded from Google Drive and placed in 'AMLS_19-20_SN16057914' folder. This is to enable the dlib feature extraction.

Google Drive link: 
Datasets: https://drive.google.com/drive/folders/15ADJ6PsgmQ2qc5iaZQJ6U4W749sMAE2E?usp=sharing
shape predictor: https://drive.google.com/file/d/1xbPWly2Vi70dsHkrQ9_0WyLNGTEpQAbA/view?usp=sharing
My name is not on the drive. Download 'Datasets' as a folder and replace the dummy Datasets folder currently in this Github. Then download the .dat file and place into the 'AMLS_19-20_SN16057914'.

My development of this code was done on Jupyter Notebook, and so there is not much commit history here to display my progress throughout the time I have been working on it. I could provide the Notebooks upon request.

Description of files in folder: 
As my models are shared between tasks A, there is only one A.py file. This is also the case for tasks B. I also have different python files to preprocess for tasks A and B. 

In the main.py code, it calls each of the python files in the folder as required. Only the preprocessing required for each task, the training, validation and testing of the final selected models are not commented out and will result in final training and testing accuracies to be printed out for each model when ran. 

There are optional code blocks that may also be ran to test my implementation. I have left these in to show my testing of different models, different preprocessing and how to plot the graphs that I have put into my report. As it would take a long time to run everything at once, I have commented them out. They are all commented clearly and should be easily understood. 

Role of each file:
Datasets = raw data used for this project, named as they are referred to in the code.
A.py = functions to build & train, validate, test and obtain cross validation scores for the model used in tasks A.
B.py = functions to build & train, validate, test and obtain cross validation scores for the model used in tasks B.
data_preprocess_A = preprocessing functions that are required for tasks A
data_preprocess_B = preprocessing functions that are required for tasks B
featureext = functions to carry out feature extraction
shape_predictor_68_face_landmarks.dat = to carry out dlib feature extraction
plots.py  = functions to plot a confusion matrix and learning curve

Packages required:
sklearn, os, numpy, keras, skimage, dlib, cv2, seaborn, matplotlib

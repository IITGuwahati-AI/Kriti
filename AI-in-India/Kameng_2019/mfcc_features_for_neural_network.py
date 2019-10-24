# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:06:55 2019

@author: tatras
"""

import wavio
import pandas as pd
import os
import numpy as np
import librosa
path = "/home/pc/file/"
train = pd.read_csv(path+"train_audio.csv")
test = pd.read_csv(path+"test_audio.csv")

    
def parser(row, file):
   file_name = os.path.join(os.path.abspath(path), file, str(row.ID) + '.wav')
   try:
       X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
       mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
       print("Error encountered while parsing file: ", file_name)
       return None, None
    
   data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
   mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,axis=0) 
   feature = mfccs   
   try:
       labels = row.Class
   except Exception as e:
       print("ID not found since its a test sample")
       return [feature], print(file_name)
   return [feature, labels], print(file_name)
    
def get_test_values(test):
    temp = test[].apply(parser, args=["Test"], axis=1)
    temp = np.array(temp).tolist()
    temp_features = []   
    for i in temp:
        temp_features.append(i[0])
    temp_features = np.array(temp_features)
    temp_features = np.squeeze(temp_features)    
    temp_features= pd.DataFrame(temp_features)
    temp_features.dropna(inplace=True)
    temp_features = np.array(temp_features).tolist()
    features = []
    for i in temp_features:
        features.append(i)
    features= np.array(features)
    return features

def get_train_values(train):
    temp = train[].apply(parser, args=["Train"], axis=1)
    temp = np.array(temp).tolist()
    temp_features = []    
    for i in temp:
        temp_features.append(i[0])
    temp_features = np.array(temp_features)
    temp_features= pd.DataFrame(temp_features)
    temp_features.dropna(inplace=True)
    temp_features = np.array(temp_features).tolist()
    temp_features = np.squeeze(temp_features)
    features = []
    classes = []    
    for i in temp_features:
        features.append(i[0])
        classes.append(i[1])     
    Xf = np.array(features)
    Yf = np.array(classes)
    return Xf, Yf

def train_model(Xtrain=None, Ytrain=None, Xtest=None, acti_lay1=None, acti_lay2=None, acti_out=None, hl1_shape=None, hl2_shape=None, batch=None, epoch=None):
    import keras
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()    
    Y2 = np_utils.to_categorical(lb.fit_transform(Ytrain))
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hl1_shape, input_shape=(Xtrain.shape[1],)))
    model.add(keras.layers.Activation(acti_lay1))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(hl2_shape))
    model.add(keras.layers.Activation(acti_lay2))
    model.add(keras.layers.Dense(Y2.shape[1]))
    model.add(keras.layers.Activation(acti_out))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
    history = model.fit(Xtrain, Y2, batch_size=batch, epochs=epoch)    
    score = model.evaluate(Xtrain, Y2)        
    prediction = model.predict(X_test)    
    return history, score, prediction



def submission(X_test=None, prediction=None):
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()     
    pred_labels = np.zeros((X_test.shape[0], ))    
    for i in range(X_test.shape[0]):
        pred_labels[i] = np.argmax(prediction[i, :])    
        pred_labels = pred_labels.astype(int)
    lb.fit(Y_train)
    labels = lb.inverse_transform(pred_labels)
    Y_test = pd.DataFrame(columns = ["ID", "Class"])  
    Y_test["ID"] = train["ID"]  
    Y_test["Class"] = pd.DataFrame(labels)    
    Y_test.to_csv(path+"submission.csv")    
    return Y_test


X_train, Y_train = get_train_values(train)
X_test = get_test_values(test)
history, score, prediction = train_model(X_train, Y_train, X_test, "sigmoid", "sigmoid", "softmax", 256, 128, 32, 500)
Y_test = submission(X_test, prediction)




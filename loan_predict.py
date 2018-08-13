# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:57:30 2018

@author: Narayanan Abishek
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def get_data():
    
    path1 = r'D:\Projects\Projects\Loan Prediction Dataset\train_u6lujuX_CVtuZ9i.csv'
    path2 = r'D:\Projects\Projects\Loan Prediction Dataset\test_Y3wMUE5_7gLdaTN.csv'
    df = pd.read_csv(path1)
    test_df = pd.read_csv(path2)
    
    gender_cat1 = pd.get_dummies(df.Gender,prefix='gender').gender_Female
    married_cat1 = pd.get_dummies(df.Married,prefix='married').married_Yes
    education_cat1 = pd.get_dummies(df.Education,prefix='education').education_Graduate
    self_employed_cat1 = pd.get_dummies(df.Self_Employed,prefix='self_employed').self_employed_Yes
    property_cat1 = pd.get_dummies(df.Property_Area,prefix='property_area')
    loan_status1 = pd.get_dummies(df.Loan_Status,prefix='loan_status').loan_status_Y
    
    gender_cat2 = pd.get_dummies(test_df.Gender,prefix='gender').gender_Female
    married_cat2 = pd.get_dummies(test_df.Married,prefix='married').married_Yes
    education_cat2 = pd.get_dummies(test_df.Education,prefix='education').education_Graduate
    self_employed_cat2 = pd.get_dummies(test_df.Self_Employed,prefix='self_employed').self_employed_Yes
    property_cat2 = pd.get_dummies(test_df.Property_Area,prefix='property_area')
    
    trainData = pd.concat([df,gender_cat1,married_cat1,education_cat1,self_employed_cat1,loan_status1,property_cat1],axis=1)
    
    testData = pd.concat([test_df,gender_cat2,married_cat2,education_cat2,self_employed_cat2,property_cat2],axis=1)

    trainData.Credit_History.fillna(trainData.Credit_History.max(),inplace=True)
    trainData.Loan_Amount_Term.fillna(trainData.Loan_Amount_Term.mean(),inplace=True)
    trainData.LoanAmount.fillna(trainData.LoanAmount.mean(),inplace=True)
    
    testData.Credit_History.fillna(testData.Credit_History.max(),inplace=True)
    testData.Loan_Amount_Term.fillna(testData.Loan_Amount_Term.mean(),inplace=True)
    testData.LoanAmount.fillna(testData.LoanAmount.mean(),inplace=True)
    
    features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','gender_Female','married_Yes','education_Graduate','self_employed_Yes','property_area_Rural','property_area_Semiurban','property_area_Urban']
    
    newData_train = trainData[features]
    newData_train[['ApplicantIncome','CoapplicantIncome','gender_Female', 'married_Yes','education_Graduate','self_employed_Yes','property_area_Rural','property_area_Semiurban','property_area_Urban']] = newData_train[['ApplicantIncome','CoapplicantIncome','gender_Female', 'married_Yes','education_Graduate','self_employed_Yes','property_area_Rural','property_area_Semiurban','property_area_Urban']].astype(float)
    
    newData_test = testData[features]
    newData_test[['ApplicantIncome','CoapplicantIncome','gender_Female', 'married_Yes','education_Graduate','self_employed_Yes','property_area_Rural','property_area_Semiurban','property_area_Urban']] = newData_test[['ApplicantIncome','CoapplicantIncome','gender_Female', 'married_Yes','education_Graduate','self_employed_Yes','property_area_Rural','property_area_Semiurban','property_area_Urban']].astype(float)

    trainData['loan_status_Y'] = trainData['loan_status_Y'].astype(float)
    loan_output_status = trainData['loan_status_Y']
    
    return newData_train,loan_output_status,newData_test

def feature_scaling(x):
    
    X_mean = np.mean(x,axis=0)
    X_sigma = np.std(x,axis=0)
    #print(X_sigma)
    X = np.divide((x - X_mean),X_sigma)
    return X

def sigmoid(Z):
    
    return 1/(1+tf.exp(-Z))

def initialize_params_linear():

    w = tf.Variable(tf.truncated_normal([90, 1], mean=0.0, stddev=1.0, dtype=tf.float64))
    b = tf.Variable(tf.zeros(1, dtype = tf.float64))
    return w,b

def calc(X,Y,w,b):
    
    predictions = sigmoid(tf.add(tf.matmul(X,w),b))   
    cost = -((Y*tf.log(predictions)) + ((1-Y)*tf.log(1-predictions)))
    error = tf.reduce_mean(cost)
    return predictions,error

def logit_regression(train_X,train_y,test_X,w,b,num_epochs=1000,learning_rate=0.01):
    
    
    predictions,cost = calc(train_X,train_y,w,b)
    points = [[],[]]
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
    #initialize the optimizer before the global initialization, since optimizer creates variables leads in memory leak
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        for i in list(range(num_epochs)):
           
            
            sess.run(optimizer)
            if i%10 == 0:
                points[0].append(i+1) #iteration
                points[1].append(sess.run(cost)) #cost
                
        pred = predictions.eval() #to convert predictions <tensor> into pred <numpy array>
            
        pred[pred>=0.5]=1.0
        pred[pred<0.5]=0.0
                     
        score_train = f1_score(train_y,pred)
        print("Training F1-score is: ",score_train)
       
        
        plt.plot(points[0],points[1], color = 'red', label = 'Train data loss')
        plt.legend()
        plt.show()
        
        
def model():
    
    w,b = initialize_params_linear()
    X,Y,test_data = get_data()
    X = np.reshape(np.array(X),(X.shape[0],X.shape[1]))
    Y = np.reshape(np.array(Y),(Y.shape[0],1))
    
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=2)
    
    my_X = poly.fit_transform(X)
       
    test_data = np.reshape(np.array(test_data),(test_data.shape[0],test_data.shape[1]))  
    
    my_test = poly.fit_transform(test_data)
    
    train_X = np.delete(my_X, 0, axis=1)
    final_train_X = feature_scaling(train_X)
    
    test_X = np.delete(my_test, 0, axis=1)
    final_test_X = feature_scaling(test_X)

    print("Shape of train-X-data:",final_train_X.shape)
    print("Shape of train-Y-data:",Y.shape)

   
    print("Shape of test-X-data:", final_test_X.shape)
    
    logit_regression(final_train_X,Y,final_test_X,w,b)
    
model()
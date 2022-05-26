#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:45:25 2020

@author: kunwang
"""

# Imports

from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.meta import OnlineSMOTEBaggingClassifier
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time



# load dataset
def load_data(file_path):
    
    data = pd.read_csv(file_path)    
    words = data['x']
    
    tfidf_v = TfidfVectorizer(max_features = 500)
    x = tfidf_v.fit_transform(words).toarray()
    
    dataset = pd.DataFrame(x)
    dataset['500'] = data['y']
    
    return dataset


def ARF_run (batch, seeds, path, data_name):
    np.random.seed(seeds)
    data = load_data(path)
    print(data.shape)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = AdaptiveRandomForestClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:])
    recall = recall_score(Y[batch:], pred[batch:])
    precision = precision_score(Y[batch:], pred[batch:])

    print("acc:", acc)
    print("f1:", f1)
    print("recall:", recall)
    print("precision:", precision)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(str(data_name)+'ARF'+str(seeds)+'.out', result, delimiter=',')
    
    return acc, f1, recall, precision
    
    
    
def NSE_run (batch, seeds, path, data_name):
    np.random.seed(seeds)
    data = load_data(path)
    print(data.shape)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = LearnPPNSEClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:])
    recall = recall_score(Y[batch:], pred[batch:])
    precision = precision_score(Y[batch:], pred[batch:])
    
    print("acc:", acc)
    print("f1:", f1)
    print("recall:", recall)
    print("precision:", precision)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(str(data_name)+'NSE'+str(seeds)+ '.out', result, delimiter=',')
    
    return acc, f1, recall, precision
    
    
    
def LEV_run ( batch, seeds, path, data_name):
    np.random.seed(seeds)
    data = load_data(path)
    print(data.shape)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = LeveragingBaggingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:])
    recall = recall_score(Y[batch:], pred[batch:])
    precision = precision_score(Y[batch:], pred[batch:])
    
    print("acc:", acc)
    print("f1:", f1)
    print("recall:", recall)
    print("precision:", precision)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(str(data_name)+'LEV'+str(seeds)+ '.out', result, delimiter=',')
    
    return acc, f1, recall, precision
    

    
def DWM_run (batch, seeds, path, data_name):
    np.random.seed(seeds)
    data = load_data(path)
    print(data.shape)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model1 = DynamicWeightedMajorityClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model1.predict(X)
        pred = np.hstack((pred,y_pred))
        model1.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:])
    recall = recall_score(Y[batch:], pred[batch:])
    precision = precision_score(Y[batch:], pred[batch:])
    
    print("acc:", acc)
    print("f1:", f1)
    print("recall:", recall)
    print("precision:", precision)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(str(data_name)+'DWM'+str(seeds)+ '.out', result, delimiter=',')
    
    return acc, f1, recall, precision
    
    
    
def OZA_run (batch, seeds, path, data_name):
    np.random.seed(seeds)
    data = load_data(path)
    print(data.shape)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = OzaBaggingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:])
    recall = recall_score(Y[batch:], pred[batch:])
    precision = precision_score(Y[batch:], pred[batch:])
    
    print("acc:", acc)
    print("f1:", f1)
    print("recall:", recall)
    print("precision:", precision)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(str(data_name)+'OZA'+str(seeds)+ '.out', result, delimiter=',')
    
    return acc, f1, recall, precision
    
    
    
    
if __name__ == '__main__':
    
    
    
    
    path = ["/home/kunwang/Data/DSS/review data/dataset name.csv"]
      
    
    data_name = ["dataset name"]
    

    
    batch_size = 50
    

    
    for i in range(len(path)):
        
        acc_total = []
        f1_total = []
        recall_total = []
        precision_total = []
        time_total = []
        
        for j in range (0, 5):
    
            # run ARF
            print (batch_size, j, 'ARF', path[i], data_name[i])
            time_start = time.time()
            acc, f1, recall, precision = ARF_run(batch_size, j, path[i], data_name[i])
            time_end = time.time()
            time_cost = time_end - time_start
            print('time_cost:', time_cost, 's')
            
            # run NSE
            print (batch_size, j, 'NSE', path[i], data_name[i])
            time_start = time.time()
            acc, f1, recall, precision = NSE_run(batch_size, j, path[i], data_name[i])
            time_end = time.time()
            time_cost = time_end - time_start
            print('time_cost:', time_cost, 's')
            
            # run LEV
            print (batch_size, j, 'LEV', path[i], data_name[i])
            time_start = time.time()
            acc, f1, recall, precision = LEV_run(batch_size, j, path[i], data_name[i])
            time_end = time.time()
            time_cost = time_end - time_start
            print('time_cost:', time_cost, 's')
            
            # run DWM
            print (batch_size, j, 'DWM', path[i], data_name[i])
            time_start = time.time()
            acc, f1, recall, precision = DWM_run(batch_size, j, path[i], data_name[i])
            time_end = time.time()
            time_cost = time_end - time_start
            print('time_cost:', time_cost, 's')
            
            # run OZA
            print (batch_size, j, 'OZA', path[i], data_name[i])
            time_start = time.time()
            acc, f1, recall, precision = OZA_run(batch_size, j, path[i], data_name[i]) 
            time_end = time.time()
            time_cost = time_end - time_start
            print('time_cost:', time_cost, 's')
            
            
            acc_total.append(acc)
            f1_total.append(f1)
            recall_total.append(recall)
            precision_total.append(precision)
            time_total.append(time_cost)
            
        
        print('-----------------------------------------------------------')
        print('acc_ave:', np.mean(acc_total), 'acc_std:', np.std(acc_total))
        print('f1_ave:', np.mean(f1_total), 'f1_std:', np.std(f1_total))
        print('recall_ave:', np.mean(recall_total), 'recall_std:', np.std(recall_total))
        print('precision_ave:', np.mean(precision_total), 'precision_std:', np.std(precision_total))
        print('time_ave:', np.mean(time_total), 'time_std:', np.std(time_total))
        print('-----------------------------------------------------------')
        
        
        
        
        
        
        
        
        
 

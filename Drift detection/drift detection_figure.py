# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:41:25 2022

@author: Kun Wang
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score




# load data
def load_data(file_path):
    
    data = pd.read_csv(file_path)    
    words = data['x']
    y = data['y'].values
    
    return words, y



def drift_detection(file_path, name):

    words, y = load_data(file_path)
    
    
    # data split
    ini_train_size = 50
    win_size = 50
    
    
    # transfer words into vertors for LDA learning
    vectorizer = CountVectorizer(max_features = 500)
    cntTf = vectorizer.fit_transform(words)
    # print (cntTf.toarray())
    
    
    # initial train set
    words_train = cntTf[0:ini_train_size, :]
    y_train = y[0:ini_train_size]
    
    
    # load LDA model
    lda = LatentDirichletAllocation(n_components = 10, random_state = 0)
    docres1 = lda.fit_transform(words_train)
    d1 = np.mean(docres1, axis = 0)

    
    # k-fold
    kf = KFold(int((cntTf.shape[0] - ini_train_size) / win_size))
    stream = cntTf[ini_train_size:, :]
    y_stream = y[ini_train_size:]
    
    
    n_virtual_drift = 0
    n_real_drift = 0
    n_normal = 0
    
    
    # transfer words intp data for model training
    tfidf_v = TfidfVectorizer(max_features = 500)
    x = tfidf_v.fit_transform(words).toarray()
    x_stream = x[ini_train_size:, :]
    
    
    # initial train set (for Bagging)
    x_train = x[0:ini_train_size, :]
    y_train = y[0:ini_train_size]
    
    
    # train decision tree
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    
    # test decision tree model
    y_pred_ini = clf.predict(x_train)


    # calculate the accuracy of decision tree model
    acc_ini = accuracy_score(y_train, y_pred_ini.T)
    
    a_count = []
    
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    y7 = []
    y8 = []
    y9 = []
    
    for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
            
        words_test = stream[test_index, :]
        y_test = y_stream[test_index]
        x_test = x_stream[test_index, :]
        
        
        # get topics for some given samples
        docres2 = lda.transform(words_test)
        d2 = np.mean(docres2, axis = 0)
        
        
        # collect the mean probability for figure plot
        y0.append(d2[0])
        y1.append(d2[1])
        y2.append(d2[2])
        y3.append(d2[3])
        y4.append(d2[4])
        y5.append(d2[5])
        y6.append(d2[6])
        y7.append(d2[7])
        y8.append(d2[8])
        y9.append(d2[9])
        
        # K-S test
        p_value = np.empty((docres1.shape[1]))
        for i in range (docres1.shape[1]):
            statistic, p = stats.ks_2samp(docres1[:, i], docres2[:, i])
            p_value[i] = p
    
    
        # drift detection
        # a = np.argwhere(p_value < 0.1)
        a = np.argwhere(p_value < 0.1)
        # print(a)
        a_count.append(a.shape[0])
        
        
        # test decision tree model
        y_pred = clf.predict(x_test)
        
        
        # calculate the accuracy of decision tree model
        acc = accuracy_score(y_test, y_pred.T)
    
        
        # count the number of drift
        if a.shape[0] >= 5:
            
            if acc >= acc_ini:
                n_real_drift += 1
                
            elif acc < acc_ini:
                n_virtual_drift += 1
            
            # retrain the LDA model
            lda = LatentDirichletAllocation(n_components = 10, random_state = 0)
            lda.fit_transform(words_test)
            
        else:
            n_normal += 1
        
        docres1 = docres2
        acc_ini = acc
    
    print('-----------------------------------------------------')
    print('n_real_drift:', n_real_drift, 'n_virtual_drift:', n_virtual_drift, 'n_normal:', n_normal)
    print(a_count)
    print('-----------------------------------------------------')
    
    
    plt.figure(figsize=(5.5,2.5))
    plt.bar(np.arange(len(a_count)), a_count)
    plt.axhline(5, linestyle = '--', color = 'r', label = '$\delta$=5, drift drigger')
    plt.xlabel('Time point')
    plt.ylabel('Number of drifted topics')
    # plt.xlim(0, 25)
    plt.ylim(0, 10)
    plt.legend(loc = 'upper right')
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.2)
    plt.savefig(str(name)+'.pdf')
    plt.show()


    # draw hybird bar chart
    y0 = np.array(y0)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    y5 = np.array(y5)
    y6 = np.array(y6)
    y7 = np.array(y7)
    y8 = np.array(y8)
    y9 = np.array(y9)
    
    labels = np.arange(len(y0))
    
    plt.figure(figsize=(5.5,3.5))
    plt.bar(labels, y0, label='Topic 1')
    plt.bar(labels, y1, bottom=y0, label='Topic 2')
    plt.bar(labels, y2, bottom=y0+y1, label='Topic 3')
    plt.bar(labels, y3, bottom=y0+y1+y2, label='Topic 4')
    plt.bar(labels, y4, bottom=y0+y1+y2+y3, label='Topic 5')
    plt.bar(labels, y5, bottom=y0+y1+y2+y3+y4, label='Topic 6')
    plt.bar(labels, y6, bottom=y0+y1+y2+y3+y4+y5, label='Topic 7')
    plt.bar(labels, y7, bottom=y0+y1+y2+y3+y4+y5+y6, label='Topic 8')
    plt.bar(labels, y8, bottom=y0+y1+y2+y3+y4+y5+y6+y7, label='Topic 9')
    plt.bar(labels, y9, bottom=y0+y1+y2+y3+y4+y5+y6+y7+y8, label='Topic 10')
    
    
    # plt.title("Error Analysis")                                                       
    plt.ylabel("Weight of topics (interests)") 
    plt.xlabel("Time point")                                                            
    plt.legend(loc='lower left')                  
    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.13)                                          
    plt.savefig(str(name)+'_interest.pdf')        
    plt.show()
    



if __name__ == '__main__':
    
    
    path = ["/home/kunwang/Data/DSS/review data/dataset name.csv"]
      
    
    file_name = ["dataset name"]
    
    
    for i in range(len(path)):
        print(path[i])
        drift_detection(path[i], fig_name[i])
        
        
        
        
        
        
        
        
        
        
        
        

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:40:26 2022

@author: Kun Wang
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import time
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
import time




class Bagging(object):
    
    
    def __init__(self, max_iter = 10, sample_rate = 0.8):
        
        # parameter initialization
        self.max_iter = max_iter
        self.sample_rate = sample_rate 
        # self.learn_rate = learn_rate
        # self.max_depth = max_depth 
        self.dtrees = []
        self.dtrees_copy = []
        self.old_learners = []
        # self.new_tree_max_iter = new_tree_max_iter
        
        
        

    def fit(self, topic_index, x_train, y_train, seeds):
        
        
        np.random.seed(seeds)
        n = x_train.shape[0]
        n_sample = int(n * self.sample_rate)
        valid_acc = np.empty((self.max_iter)) 
        
        
        # get the training subset
        for iter_ in range(self.max_iter): 
            topic_sample_idx = np.where(topic_index == iter_)
            topic_sample_idx = np.array(topic_sample_idx)
            topic_sample_idx = topic_sample_idx.flatten()
            
            
            x_train_subset1, y_train_subset1 = x_train[topic_sample_idx, :], y_train[topic_sample_idx]
            
            
            other_x_train = np.delete(x_train, topic_sample_idx, axis = 0)
            other_y_train = np.delete(y_train, topic_sample_idx)
            n_other_sample = n_sample - topic_sample_idx.shape[0]
            other_sample_idx = np.random.permutation(other_x_train.shape[0])[:n_other_sample]
            
            
            x_train_subset2, y_train_subset2 = other_x_train[other_sample_idx, :], other_y_train[other_sample_idx]
            
            
            x_train_subset = np.vstack((x_train_subset1, x_train_subset2))
            y_train_subset = np.hstack((y_train_subset1, y_train_subset2))
            
            
            # get the validation set
            x_valid = np.delete(other_x_train, other_sample_idx, axis = 0)
            y_valid = np.delete(other_y_train, other_sample_idx)
            
            
            # train the weak learner
            clf = DecisionTreeClassifier(max_depth = 2)
            clf.fit(x_train_subset, y_train_subset)
            
            
            # validate the weak learner and get accuracy
            y_pred = clf.predict(x_valid)
            acc = accuracy_score(y_valid, y_pred)
            valid_acc[iter_] = acc
            
            
            # add the trained weak learner in the ensemble
            self.dtrees.append(clf)
            self.dtrees_copy = self.dtrees.copy()
            
            
        return valid_acc
            
        
            

    def predict(self, x_test, docres, valid_acc):
        
        topic_index = np.argmax(docres, axis = 1)
        topic_count = []
        
        for i in range(self.max_iter):
            n_topic = np.sum(topic_index == i)
            topic_count.append(n_topic)     
        n_test = x_test.shape[0]
        
        
        # test_score = np.empty((self.max_iter, n_test))
        test_score = np.empty((len(self.dtrees), n_test))
        pred_score = np.empty((n_test,)) 
        
        
        # integrate the results of weak learners    
        for i in range(len(self.dtrees)):
            test_score[i, :] = self.dtrees[i].predict(x_test)
        test_score = test_score.T
            
              
        
        # # majority voting
        # for i in range (n_test):
        #     pred_score[i] = np.mean(test_score[i,:])
            
        
        # acc weighted voting
        for i in range (n_test):
            pred_score[i] = np.dot(test_score[i,:], valid_acc)
            pred_score[i] = pred_score[i] / np.sum(valid_acc)
            
            
        # construct new training data for GBDT
        topic_index_array = topic_index.reshape((test_score.shape[0], 1))
        label_topic = topic_index_array

        return pred_score.reshape(-1, 1), test_score, label_topic
    
    
    
    def learner_selection(self, y_test, test_score, valid_acc):
        

        self.dtrees_copy = self.dtrees.copy()
        delete = []
        keep = []
        
        # get the accuracy of each learner
        test_acc = np.empty((self.max_iter)) 
        for i in range (test_score.shape[1]):
            test_acc[i] = accuracy_score(y_test, test_score[:,i])
        
        
        # compare the test_acc and initial valid_acc
        for i in range (self.max_iter):
            if test_acc[i] <= valid_acc[i]:
                delete.append(i)
            else:
                keep.append(i)
        
        
        # choose poor learner index
        del_index = np.array(delete)
        del_index = del_index.flatten()
        # print(del_index)
        
        keep_index = np.array(keep)
        keep_index = keep_index.flatten()
        
        # print('del_index:', del_index)
        
        
        return del_index, keep_index, test_acc

    
    
    
    def fit_new_learner(self, topic_index_test, del_index, x_train, y_train, test_acc, seeds):
        
        
        np.random.seed(seeds)
        n = x_train.shape[0]
        n_sample = int(n * self.sample_rate)
        update_test_acc = test_acc
        
        
        # get the training subset
        for iter_ in range(del_index.shape[0]): 
            topic_sample_idx = np.where(topic_index == del_index[iter_])
            topic_sample_idx = np.array(topic_sample_idx)
            topic_sample_idx = topic_sample_idx.flatten()
            
            
            x_train_subset1, y_train_subset1 = x_train[topic_sample_idx, :], y_train[topic_sample_idx]
            
            
            other_x_train = np.delete(x_train, topic_sample_idx, axis = 0)
            other_y_train = np.delete(y_train, topic_sample_idx)
            n_other_sample = n_sample - topic_sample_idx.shape[0]
            other_sample_idx = np.random.permutation(other_x_train.shape[0])[:n_other_sample]
            
            
            x_train_subset2, y_train_subset2 = other_x_train[other_sample_idx, :], other_y_train[other_sample_idx]
            
            
            x_train_subset = np.vstack((x_train_subset1, x_train_subset2))
            y_train_subset = np.hstack((y_train_subset1, y_train_subset2))
            
            
            # get the validation set
            x_valid = np.delete(other_x_train, other_sample_idx, axis = 0)
            y_valid = np.delete(other_y_train, other_sample_idx)
            
            
            # train the weak learner
            clf = DecisionTreeClassifier(max_depth = 2)
            clf.fit(x_train_subset, y_train_subset)
            
            
            # validate the weak learner and get accuracy
            y_pred = clf.predict(x_valid)
            update_test_acc[del_index[iter_]] = accuracy_score(y_valid, y_pred)

            
            # replace the trained weak learner in the ensemble
            self.dtrees[del_index[iter_]] = clf
            
        # print(update_test_acc)
        return update_test_acc
        
        
        

# load data
def load_data(file_path):
    
    data = pd.read_csv(file_path)    
    words = data['x']
    y = data['y'].values
    
    return words, y



def data_transfer(words, y):
    
    # transfer words into vertors
    vectorizer = CountVectorizer(max_features = 500)
    cntTf = vectorizer.fit_transform(words)
    # print (cntTf.toarray())
    
    # transfer words into data
    tfidf_v = TfidfVectorizer(max_features = 500)
    x = tfidf_v.fit_transform(words).toarray()
    y = y
    
    return cntTf, x, y


    
if __name__ == '__main__':
    
    
    # load dataset
    path = ["/home/kunwang/Data/DSS/review data/dataset name.csv"]
      
    
    path_name = ["dataset name"]
    
    
    
    for data_id in range(len(path)):
        
        acc_total = []
        f1_total = []
        recall_total = []
        precision_total = []
        time_total = []
        
        for j in range(0, 1):
           
            time_start = time.time()
            
            print('------------', data_id)
            words, y = load_data(path[data_id])
            print(path[data_id], 'seeds:', j)
        
        
            # data split
            ini_train_size = 50
            win_size = 50
            
            
            # data transfer
            cntTf, x, y = data_transfer(words, y)
            
            
            # initial train set
            words_train = cntTf[0:ini_train_size, :]  # for LDA
            x_train = x[0:ini_train_size, :]  # for Bagging
            y_train = y[0:ini_train_size]  # for Bagging
            
            
            # Topic extract, load LDA model
            lda = LatentDirichletAllocation(n_components = 10, random_state = 0)
            docres1 = lda.fit_transform(words_train)
            topic_index = np.argmax(docres1, axis = 1)
            
            
            # load and train model
            model = Bagging()
            valid_acc = model.fit(topic_index, x_train, y_train, j)
            
            
            # construct data and train a GBDT model
            y_residual_ini, test_score_ini, label_topic_ini = model.predict(x_train, docres1, valid_acc)
            y_train_gb = y_train
            GBDT = GradientBoostingClassifier(n_estimators=10, subsample=0.8)
            GBDT.fit(label_topic_ini, y_train_gb)
            
            
            # k-fold
            kf = KFold(int((x.shape[0] - ini_train_size) / win_size))
            stream_lda = cntTf[ini_train_size:, :]
            stream = x[ini_train_size:, :]
            pred = np.zeros(stream.shape[0])
            batch_acc = []
            
            y_pred_cum = np.empty(0)
            y_test_cum = np.empty(0)
            
            
            for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
                
                
                words_test = stream_lda[test_index, :]
                x_test = stream[test_index, :]
                y_test = y[test_index]
                
                
                # get topics for some given samples
                docres2 = lda.transform(words_test)
                topic_index_test = np.argmax(docres2, axis = 1)
                
                
                # K-S test
                p_value = np.empty((docres1.shape[1]))
                for i in range (docres1.shape[1]):
                    statistic, p = stats.ks_2samp(docres1[:, i], docres2[:, i])
                    p_value[i] = p
                    
                
                # Bagging model results
                y_residual, test_score, label_topic = model.predict(x_test, docres2, valid_acc)
                y_residual = y_residual.flatten()            
                y_pred = (y_residual >= 0.5)
                
                
                # GBDT model results
                y_pred_gb = GBDT.predict(label_topic)
                
                acc_bag = metrics.accuracy_score(y_test, y_pred.T)
                acc_gb = metrics.accuracy_score(y_test, y_pred_gb)
                
                
                # adaptive weighted voting
                # (1) get the acc of test_score and y_pred_GB
                gb_valid_acc = np.empty((10)) 
                pred_score2 = np.empty((test_score.shape[0]))
                for i in range (10):
                    gb_valid_acc[i] = accuracy_score(y_pred_gb, test_score[:, i])
                
                # print('gb_valid_acc', gb_valid_acc)
                ix = np.array(np.where(gb_valid_acc == np.max(gb_valid_acc)))
                ix = ix.flatten()
    
                    
                # (2) voting
                for i in range (test_score.shape[0]):
                    pred_score2[i] = np.mean(test_score[i,ix])
                
                    
                pred_score2 = pred_score2.flatten()
                y_pred2 = (pred_score2 >= 0.5)    
                
                acc_weight = accuracy_score(y_test, y_pred2)
      
                batch_acc.append(np.max([acc_bag, acc_gb, acc_weight]))
                acc_idx = np.argmax([acc_bag, acc_gb, acc_weight])
                
                
                # # (3) select the best result
                # if acc_idx == 0:
                #     y_pred_cum = np.hstack((y_pred_cum, y_pred))               
                # elif acc_idx == 1:
                #     y_pred_cum = np.hstack((y_pred_cum, y_pred_gb))               
                # elif acc_idx == 2:
                #     y_pred_cum = np.hstack((y_pred_cum, y_pred2)) 
                    
                y_pred_cum = np.hstack((y_pred_cum, y_pred))
                y_test_cum = np.hstack((y_test_cum, y_test))
                    
                
                #----------------------------------------
                # interest drift detection based learning 
                #----------------------------------------
                
                a = np.argwhere(p_value < 0.1)
                
                
                if a.shape[0] >= 5:  
                    model = Bagging()
                    update_test_acc = model.fit(topic_index_test, x_test, y_test, j)
                    
                    label_topic_ini = label_topic
                    y_train_gb = y_test
                    
                    GBDT = GradientBoostingClassifier(n_estimators=10, subsample=0.8)
                    GBDT.fit(label_topic_ini, y_train_gb)    
                    
                else:
                    del_index, keep_index, test_acc = model.learner_selection(y_test, test_score, valid_acc) 
                    update_test_acc = model.fit_new_learner(topic_index_test, del_index, x_test, y_test, test_acc, j)
                
                    label_topic_ini = np.vstack((label_topic_ini, label_topic))
                    y_train_gb = np.hstack((y_train_gb, y_test))
                
                    GBDT = GradientBoostingClassifier(n_estimators=10, subsample=0.8)
                    GBDT.fit(label_topic_ini, y_train_gb)
               
                
                docres1 = docres2
                valid_acc = update_test_acc
                 
            time_end = time.time()
                

            acc = accuracy_score(y_test_cum, y_pred_cum)
            f1 = f1_score(y_test_cum, y_pred_cum)
            recall = recall_score(y_test_cum, y_pred_cum)
            precision = precision_score(y_test_cum, y_pred_cum)
            
            
            acc_total.append(acc)
            f1_total.append(f1)
            recall_total.append(recall)
            precision_total.append(precision)
            
            time_cost = time_end - time_start
            time_total.append(time_cost)
            
            
            print('acc:', acc)
            print('f1:', f1)
            print('recall:', recall)
            print('precision:', precision)
            print('time_cost:', time_cost, 's')
            
            
            # save results
            result = np.zeros([y_pred_cum.shape[0], 2])
            result[:, 0] = y_pred_cum
            result[:, 1] = y_test_cum
            np.savetxt(str(path_name[data_id])+'seeds'+ str(j) + 'es_acc.out', result, delimiter=',')
    
                    
        # print('-----------------------------------------------------------')
        # print('acc_ave:', np.mean(acc_total), 'acc_std:', np.std(acc_total))
        # print('f1_ave:', np.mean(f1_total), 'f1_std:', np.std(f1_total))
        # print('recall_ave:', np.mean(recall_total), 'recall_std:', np.std(recall_total))
        # print('precision_ave:', np.mean(precision_total), 'precision_std:', np.std(precision_total))
        # print('time_ave:', np.mean(time_total), 'time_std:', np.std(time_total))
        # print('-----------------------------------------------------------')

    
    

    
    











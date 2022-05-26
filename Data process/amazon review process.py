# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:29:06 2021

@author: kunwang
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re



input_path_name = ["/home/kunwang/Data/DSS/review data/dataset name.csv"]

output_path_name = ["/home/kunwang/Data/DSS/review data process/dataset name.csv"]


n = len(input_path_name)

for n in range(n):
    
    print(input_path_name[n])
    print(output_path_name[n])
    
    df = pd.read_csv(input_path_name[n])
    
    # feature pre-process
    df.drop(df[df['review text'].isna()].index, inplace=True) # drop where there are no text
    df.reset_index(drop = True, inplace=True)
    
    # data preprocessing
    ps = PorterStemmer()
    corpus = []
    
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df.loc[i, 'review text'])
        review = review.lower()
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    df['review text'] = corpus   
    words = df['review text']
    y = df['class']
    
    # save data to csv file
    data = pd.DataFrame(columns=['x','y'])
    data['x'] = words
    data['y'] = y
    
    data = data[~(data['x'] == '')] 
    data.reset_index(drop = True, inplace=True)
    
    data.to_csv(output_path_name[n], index=False, header=1)






# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:16:01 2022

@author: Administrator
"""
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


output_path_name = ["/home/kunwang/Data/DSS/review data binary/dataset name.csv"]


n = len(input_path_name)

for n in range(n):
    
    print(input_path_name[n])
    print(output_path_name[n])
    
    data = pd.read_csv(input_path_name[n])
    
    # multiclasses to binary classes
    for i in range(0, len(data)):
        
        if data['y'][i] <= 3:
            data['y'][i] = 0
        else:
            data['y'][i] = 1
            
    
    # save data
    data.to_csv(output_path_name[n], index=False, header=1)










#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:52:57 2018

@author: aarushi
"""

import pandas as pd
import numpy as np

data = pd.read_csv('./data/data/ConceptDetectionTraining2018-Concepts.csv', sep='\t', header=None)
print(data.shape)
data.reset_index(inplace=True)
concepts = {}
# for i, row in data.iteritems():
for i in range(data.shape[0]):
    row = data.loc[i]
    try:
        lst = np.array(row[1].split(';'))
        for concept in lst:
            concepts[concept] = 0
    except:
        print(row[1])
    if (i % 10000) == 0:
        print(i)
print("number of different concepts is {}".format(len(concepts.keys())))
# number of different concepts is 111156

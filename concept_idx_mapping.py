#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:20:01 2018

@author: aarushi
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
from utils import *

data_train = pd.read_csv('./data/data/ConceptDetectionTraining2018-Concepts.csv', sep='\t', header=None)
image_list = np.load('./data/data/train_image_list_concept.npy')

print(len(image_list))
print(data_train.shape)

image_concept = {}
for i, row in data_train.iterrows():
    image_concept[row[0] + '.jpg'] = row[1]
print(len(list(image_concept.keys())))

concepts = {}
for image in image_list:
    try:
        concept = image_concept[image.decode('UTF-8')]
        concept_list = concept.split(';')
        for conc in concept_list:
            if conc in concepts:
                concepts[conc] += 1
            else:
                concepts[conc] = 1
    except:
        continue

print(len(list(concepts.keys())))
#    177851
# chosing 18.6k most frequent concepts accounting for [0.92170681639004348] of total concepts utilisation

idx_concept = {}
concept_idx = {}
idx_freq = {}
curr_idx = 0
concepts = OrderedDict(sorted(concepts.items(), key=lambda kv: kv[1], reverse=True))
for conc, freq in concepts.items():
    idx_concept[curr_idx] = conc
    idx_freq[curr_idx] = freq
    concept_idx[conc] = curr_idx
    curr_idx += 1

save_pickle(idx_concept, "./data/idx_concept.pkl")
save_pickle(idx_freq, "./data/idx_freq.pkl")
save_pickle(concept_idx, "./data/concept_idx.pkl")

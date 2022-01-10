import pandas as pd
import numpy as np
import hickle
from utils import *

# freq_thresh=67
idx_freq = load_pickle('./data/idx_freq.pkl')
concept_idx = load_pickle('./data/concept_idx.pkl')
data_train = pd.read_csv('./data/data/ConceptDetectionTraining2018-Concepts.csv', sep='\t', header=None)
image_list = np.load('./data/data/train_image_list_concept.npy')

print('files_loaded')
image_concept = {}
for i, row in data_train.iterrows():
    image_concept[str(row[0] + '.jpg')] = row[1]
print('image_concepts read')
# idx_max=-1
# for idx,freq in idx_freq.iteritems():
#    if(freq>=freq_thresh):
#        idx_max=idx
#    else:
#        break
idx_max = 10000 - 1
print('number of concepts = ' + str(idx_max + 1))
num_images = len(image_list)
num_batches = int(num_images / 10000)

for batch in range(num_batches):
    image_concept_encoding = np.zeros((10000, idx_max + 1), dtype=np.int8)
    for i in range(10000):
        image = image_list[i + (10000 * batch)]
        concept_list = image_concept[str(image)].split(';')
        for concept_name in concept_list:
            idx = concept_idx[concept_name]
            freq = idx_freq[idx]
            #        if(freq>freq_thresh):
            if (idx <= idx_max):
                image_concept_encoding[i][idx] = 1
    print('batch: ', batch)
    hickle.dump(image_concept_encoding, './data/encoding/encoding_concept_' + str(batch) + '.hkl')
    print(batch, 5 + (10000 * batch))
print(num_images - (10000 * num_batches))
image_concept_encoding = np.zeros((num_images - (10000 * num_batches), idx_max + 1), dtype=np.int8)
for i in range(num_images - (10000 * num_batches)):
    image = image_list[i + (10000 * num_batches)]
    concept_list = image_concept[image].split(';')
    for concept_name in concept_list:
        idx = concept_idx[concept_name]
        freq = idx_freq[idx]
        #        if(freq>freq_thresh):
        if (idx <= idx_max):
            image_concept_encoding[i][idx] = 1
hickle.dump(image_concept_encoding, './data/encoding/encoding_concept_' + str(num_batches) + '.hkl')

# hickle.dump(image_concept_encoding,'./data/image_concept_encoding.hkl')
# save_pkl(image_concept_encoding,"./data/image_concept_encoding.pkl")

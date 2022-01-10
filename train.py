import numpy as np
import hickle
from model import *
import cv2
from utils import *


def get_encoding(img_idx):
    file_name = './data/encoding/' + str(int(img_idx / 1000)) + '.pkl'
    encoding = load_pkl(file_name)[img_idx % 1000]
    return encoding


def get_img_arr(img_filename):
    img = cv2.imread('./images/train_resized3/' + img_filename)
    img = np.array(img).astype(float)
    img[:, :, 0] -= 100.402736525
    img[:, :, 1] -= 90.0410865677
    img[:, :, 2] -= 90.5816390194
    img = img / 255.0
    #    img = np.expand_dims(img, axis=0)
    return img


def get_concept_encoding(i, j):
    file_name = './data/encoding/encoding_concept_' + str(int(i / 10000)) + '.hkl'
    embedding = hickle.load(file_name)
    index_start = i % 10000
    index_end = j % 10000
    if (int(i / 10000) == int(j / 10000)):
        return embedding[index_start:index_end]
    else:
        return np.concatenate((embedding[index_start:], get_concept_encoding(((i / 10000) + 1) * 10000, j)), axis=0)


if __name__ == '__main__':
    image_list = np.load('./data/train_image_list_concept.npy')
    EPOCHS = 10
    X_train = []
    #    Y_train=[]
    for i in range(100):
        img_filename = image_list[i]
        X_train.append(get_img_arr(img_filename))
    Y_train = get_concept_encoding(0, 100)
    Y_train = Y_train[:, :1000]
    BATCHSIZE = 10
    model = get_model()
    model.fit(np.array(X_train), Y_train, batch_size=BATCHSIZE, nb_epoch=EPOCHS, shuffle="batch", verbose=1)
    model.save_weights('my_model_weights1.h5')
    model.save('my_model1.h5')
    # score = model.evaluate(X_test, Y_test, batch_size=32)
    # model.fit(X_train, Y_train, batch_size=32, nb_epoch=15 ,show_accuracy=True)
    # out = model.predict(im)
    # model.save_weights('my_model_weights.h5')
    # model.save('my_model.h5')

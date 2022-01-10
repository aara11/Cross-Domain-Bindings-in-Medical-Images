import numpy as np
import os
from PIL import Image


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b: return False
    return True


direc = './images/train_resized3/'
img_list = np.array(os.listdir(direc))

np.random.shuffle(img_list)
gs_img_list = []
i = 1
for i in range(len(img_list)):
    if i % 100 == 0:
        print(i)
    img = img_list[i]
    if is_grey_scale(direc + img):
        os.system('cp ' + direc + img + ' ./images/' + img)
        gs_img_list.append(img)

gs_img_list = np.array(gs_img_list)
np.save('gs_image_list.npy', gs_img_list)

direc = './images/train_resized'
img_list = os.listdir(direc)
img_list = np.array(img_list)
np.random.shuffle(img_list)
img_list = img_list[:int(len(img_list) * .8)]
np.save('train_image_list_concept.npy', img_list)

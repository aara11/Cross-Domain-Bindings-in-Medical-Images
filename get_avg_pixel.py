from PIL import Image
import os
import numpy as np

img_list = np.load('./data/gs_image_list.npy')
direc_lst = ['./images']
# direc_lst = ['./images/train_resized2', './images/train_resized1', './images/train_resized']

for direc in direc_lst:
    r = []
    g = []
    b = []

    num_images = len(img_list)
    for i in range(num_images):

        if (i % 1000 == 0):
            print(i)
        img = img_list[i]
        image = Image.open(os.path.join(direc, img)).convert('RGB')

        temp = np.array(image)

        red_images = temp[:, :, 0]
        green_images = temp[:, :, 1]
        blue_images = temp[:, :, 2]
        r.append(np.mean(red_images))
        g.append(np.mean(green_images))
        b.append(np.mean(blue_images))

    print(np.mean(r))
    print(np.mean(g))
    print(np.mean(b))
# grey [r g b] -> 80.17094439168521 80.17094439168521 80.16999994900743
# print(str(np.mean(red_image)))
# resized2 ->  [r g b]=[100.402736525, 90.0410865677, 90.5816390194]
# resized1 ->  [r g b]=[132.798033071, 119.027250008, 119.732772281]
# resized ->  [r g b]=[135.043927868, 119.80514027, 120.3567889]

# img = cv2.resize(cv2.imread('../../Downloads/cat2.jpg'), (224, 224))
#
#    mean_pixel = [103.939, 116.779, 123.68]
#    img = img.astype(np.float32, copy=False)
#    for c in range(3):
#        img[:, :, c] = img[:, :, c] - mean_pixel[c]
#    img = img.transpose((2,0,1))
#    img = np.expand_dims(img, axis=0)
# temp

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:39:39 2018

@author: siplab
"""

from PIL import Image
import os
from PIL import ImageOps


def resize_image1(image):
    #    width, height = image.size
    #    if width > height:
    #        left = (width - height) / 2
    #        right = width - left
    #        top = 0
    #        bottom = height
    #    else:
    #        top = (height - width) / 2
    #        bottom = height - top
    #        left = 0
    #        right = width
    #    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image


def resize_image3(image):
    width, height = image.size
    desired_size = max(height, width)
    delta_w = desired_size - width
    delta_h = desired_size - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h / 2))
    image = ImageOps.expand(image, padding)
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image


def main():
    splits = ['train']  # , 'val']
    for split in splits:
        folder = './images/train'
        resized_folder1 = "./images/%s_resized1/{}".format(split)
        resized_folder3 = "./images/%s_resized3/{}".format(split)
        if not os.path.exists(resized_folder1):
            os.makedirs(resized_folder1)
        if not os.path.exists(resized_folder3):
            os.makedirs(resized_folder3)

        print('Start resizing {} images.'.format(split))
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            image = Image.open(os.path.join(folder, image_file))
            r_image1 = resize_image1(image)
            r_image1.save(os.path.join(resized_folder1, image_file), image.format)
            r_image3 = resize_image3(image)
            r_image3.save(os.path.join(resized_folder3, image_file), image.format)
            if i % 1000 == 0:
                print('Done: {}/{}'.format(i, num_images))


if __name__ == '__main__':
    main()

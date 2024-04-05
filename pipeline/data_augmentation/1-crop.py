#!/usr/bin/env python3
"""crop_image"""
from tensorflow.image import random_crop

def crop_image(image, size):
    '''crop_image'''
    return random_crop(image,size)

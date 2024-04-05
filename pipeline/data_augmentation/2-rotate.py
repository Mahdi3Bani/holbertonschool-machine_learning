#!/usr/bin/env python3
"""rot 90"""
from tensorflow.image import rot90

def rotate_image(image): 
    '''rot 90'''
    return rot90(image)

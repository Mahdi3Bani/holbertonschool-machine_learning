#!/usr/bin/env python3
"""flipping an image horizontally"""
import tensorflow as tf


def flip_image(image): 
    '''flip image horizntally'''
    return tf.image.flip_left_right(image)

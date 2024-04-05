#!/usr/bin/env python3
"""change brightness"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """change brightness"""
    return tf.image.adjust_brightness(image, max_delta)
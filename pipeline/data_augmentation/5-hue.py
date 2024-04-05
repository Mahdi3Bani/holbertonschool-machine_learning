#!/usr/bin/env python3
"""change hue"""
import tensorflow as tf


def change_hue(image, delta):
    """change hue"""
    return tf.image.adjust_hue(image, delta)
#!/usr/bin/env python3
"""random_shear"""
import tensorflow as tf


def shear_image(image, intensity):
    """Shear image"""
    return tf.keras.preprocessing.image.random_shear(image, intensity)
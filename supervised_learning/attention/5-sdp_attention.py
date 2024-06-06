#!/usr/bin/env python3
import tensorflow as tf
"""
sdp attention
"""


def sdp_attention(Q, K, V, mask=None):
    """
    sdp attention
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
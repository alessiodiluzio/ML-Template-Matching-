import tensorflow as tf


def compute_cross_entropy_loss(logits, label, balance_factor, training=True):
    cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    if training:
        class_weights = tf.constant([[[[1.0 / balance_factor, 1.0 / balance_factor]]]])
        weights = tf.reduce_sum(class_weights * label, axis=-1)
        weighted_loss = tf.reduce_mean(cross_entropy * weights)
        return weighted_loss
    return cross_entropy

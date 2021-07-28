import tensorflow as tf


def compute_cross_entropy_loss(logits, label, balance_factor, training=True):
    cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    if training:
        class_weights = tf.constant([[[[1.0 / balance_factor, 1.0 / balance_factor]]]])
        weights = tf.reduce_sum(class_weights * label, axis=-1)
        weighted_loss = tf.reduce_mean(cross_entropy * weights)
        return weighted_loss
    return cross_entropy


def logistic_loss(logits, label, balance_factor, training=True):
    weights = 1.0
    if training:
        label_true = (label + 1) / 2
        label_false = (label - 1) / 2
        weights = 1 / balance_factor * label_true + 1 / (1 - balance_factor) * label_false
    log_loss = tf.compat.v1.losses.log_loss(labels=label, predictions=logits, weights=weights,
                                            reduction='none')
    log_loss = tf.reduce_mean(log_loss)
    return log_loss

import tensorflow as tf


def get_balanced_weigths(balance_factor, label):
    label_true = (label + 1) / 2
    label_false = (label - 1) / 2
    weights = 1 / balance_factor * label_true + 1 / (1 - balance_factor) * label_false
    return weights


def softmax_cross_entropy_loss(logits, label, balance_factor, training=True):
    cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    if training:
        cross_entropy = tf.expand_dims(cross_entropy, axis=3)
        weights = get_balanced_weigths(balance_factor, label)
        cross_entropy = tf.math.multiply(cross_entropy, weights)
        return tf.reduce_mean(cross_entropy)
    return cross_entropy


def compute_logistic_loss(labels, logits):
    loss = tf.math.multiply(logits, labels) * -1
    # l(y,v) = -y*v
    loss = tf.math.exp(loss)
    # l(y,v) = exp(-yv)
    loss = tf.math.log(1 + loss)
    # l(y, v) = log((1 + exp(-yv))
    return loss


def logistic_loss(logits, label, balance_factor, training=True):
    log_loss = compute_logistic_loss(label, logits)
    if training:
        weights = get_balanced_weigths(balance_factor, label)
        log_loss = tf.math.multiply(log_loss, weights)
    log_loss = tf.reduce_mean(log_loss)
    return log_loss


def sigmoid_cross_entropy_loss(logits, label, balance_factor, training=True):
    cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    if training:
        cross_entropy = tf.expand_dims(cross_entropy, axis=3)
        weights = get_balanced_weigths(balance_factor, label)
        cross_entropy = tf.math.multiply(cross_entropy, weights)
        return tf.reduce_mean(cross_entropy)
    return cross_entropy

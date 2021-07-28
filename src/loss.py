import tensorflow as tf


def get_balanced_weigths(balance_factor, label):
    label_true = (label + 1) / 2
    label_false = (label - 1) / 2
    weights = 1 / balance_factor * label_true + 1 / (1 - balance_factor) * label_false
    return weights


def softmax_cross_entropy_loss(logits, label, balance_factor, training=True):
    cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    cross_entropy = tf.expand_dims(cross_entropy, axis=3)
    if training:
        weights = get_balanced_weigths(balance_factor, label)
        weighted_loss = tf.reduce_mean(cross_entropy * weights)
        return weighted_loss
    return cross_entropy


def logistic_loss(logits, label, balance_factor, training=True):
    weights = 1.0
    if training:
        weights = get_balanced_weigths(balance_factor, label)
    log_loss = tf.compat.v1.losses.log_loss(labels=label, predictions=logits, weights=weights,
                                            reduction='none')
    log_loss = tf.reduce_mean(log_loss)
    return log_loss


def compute_sigmoid_cross_entropy(labels, logits):
    positive = - labels * tf.math.log(tf.nn.sigmoid(logits))
    negative = (1 - labels) * - tf.math.log(tf.nn.sigmoid(1.0 - logits))
    print(positive + negative)
    print(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    positive = tf.where(tf.equal(positive, 0), positive, positive)
    return tf.math.reduce_sum(-1 * positive, axis=-1)


def sigmoid_cross_entropy_loss(logits, label, balance_factor, training=True):
    cross_entropy = compute_sigmoid_cross_entropy(label, logits)
    cross_entropy = tf.expand_dims(cross_entropy, axis=3)
    if training:
        weights = get_balanced_weigths(balance_factor, label)
        weighted_loss = tf.reduce_mean(cross_entropy * weights)
        return weighted_loss
    return cross_entropy

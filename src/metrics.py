import tensorflow as tf


def true_positives(predictions, labels):
    return tf.math.count_nonzero(predictions * labels, dtype=tf.float32)


def true_negatives(predictions, labels):
    return tf.math.count_nonzero((predictions - 1) * (labels - 1), dtype=tf.float32)


def false_positives(predictions, labels):
    return tf.math.count_nonzero(predictions * (labels - 1), dtype=tf.float32)


def false_negatives(predictions, labels):
    return tf.math.count_nonzero((predictions - 1) * labels, dtype=tf.float32)


def accuracy(logits, labels):
    # predictions = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=-1), tf.float32)
    # predictions = tf.expand_dims(predictions, axis=3)

    tp = true_positives(logits, labels)
    tn = true_negatives(logits, labels)
    fp = false_positives(logits, labels)
    fn = false_negatives(logits, labels)

    if tf.equal((tp + tn + fn + fp), 0):
        return 1.0
    else:
        return (tp + tn) / (tp + fn + tn + fp)


def precision_recall(logits, labels):

    predictions = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=-1), tf.float32)
    predictions = tf.expand_dims(predictions, axis=3)

    tp = true_positives(predictions, labels)
    fp = false_positives(predictions, labels)
    fn = false_negatives(predictions, labels)

    if tf.equal((tp + fn + fp), 0):
        precision = tf.ones(shape=())
        recall = tf.ones(shape=())
    else:
        if tf.equal((tp + fp), 0):
            precision = tf.zeros(shape=())
        else:
            precision = tp / (tp + fp)
        if tf.equal((tp + fn), 0):
            recall = tf.zeros(shape=())
        else:
            recall = tp / (tp + fn)

    return precision, recall


def f1score(precision, recall):
    if tf.equal((precision + recall), 0):
        f1 = tf.zeros(())
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

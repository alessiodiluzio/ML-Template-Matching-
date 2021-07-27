import tensorflow as tf
import os

from src.model import Siamese
from src.metrics import precision_recall, accuracy, f1score
from src.loss import compute_cross_entropy_loss
from src.utils import save_plot, plot_metrics, get_balance_factor, get_device
from src import LEARNING_RATE
from IPython.display import clear_output


def train(training_set, validation_set, epochs, train_steps, val_steps, plot_path, image_path,
          optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), early_stopping=None):

    device = get_device()
    siam_model = Siamese(checkpoint_dir="checkpoint", device=device)

    best_f1score = 0
    last_improvement = 0

    # Initialize dictionary to store the history
    siam_model.history = {'train_loss': [], 'val_loss': [], 'train_f1score': [],
                          'val_f1score': [], 'train_acc': [], 'val_acc': []}
    balance_factor = get_balance_factor()
    epoch = 0
    for epoch in range(epochs):
        clear_output()

        print("Epoch: {}/{}".format(epoch + 1, epochs))

        train_loss = tf.metrics.Mean('train_loss')
        train_f1score = tf.metrics.Mean('train_f1')
        train_accuracy = tf.metrics.Mean('train_acc')

        train_progbar = tf.keras.utils.Progbar(train_steps)

        # TRAIN LOOP

        print("\nTRAIN")

        for b, (image, template, label) in enumerate(training_set):
            one_hot_labels = tf.one_hot(indices=label, depth=2, dtype=tf.float32)
            label = tf.cast(label, dtype=tf.float32)
            logits, loss = siam_model.forward_backward_pass([image, template], one_hot_labels, optimizer)
            precision, recall = precision_recall(logits, label)
            f1score_value = f1score(precision, recall)
            accuracy_value = accuracy(logits, label)
            train_loss(loss)
            train_f1score(f1score_value)
            train_accuracy(accuracy_value)
            metrics = [('loss', loss), ("f1", f1score_value), ("accuracy", accuracy_value)]
            train_progbar.update(b + 1, metrics)

        val_loss = tf.metrics.Mean('val_loss')
        val_f1score = tf.metrics.Mean('val_f1')
        val_accuracy = tf.metrics.Mean('val_acc')

        val_progbar = tf.keras.utils.Progbar(val_steps)

        # VALIDATION LOOP

        print("\nVALIDATE")

        for b, (image, template, label) in enumerate(validation_set):

            one_hot_labels = tf.one_hot(indices=label, depth=2, dtype=tf.float32)
            label = tf.cast(label, dtype=tf.float32)
            logits = siam_model.forward([image, template])
            loss = compute_cross_entropy_loss(logits, one_hot_labels,
                                              balance_factor=balance_factor, training=False)

            precision, recall = precision_recall(logits, label)
            f1score_value = f1score(precision, recall)
            accuracy_value = accuracy(logits, label)

            val_loss(loss)
            val_f1score(f1score_value)
            val_accuracy(accuracy_value)
            metrics = [('val_loss', loss), ("val_f1", f1score_value), ("val_acc", accuracy_value)]
            val_progbar.update(b + 1, metrics)

        i = 0
        for image, template, label in validation_set.take(3):
            prediction = siam_model.forward([image, template])
            save_plot(image[i], template[i], label[i], logit=prediction[i], dest=os.path.join(image_path, str(epoch)+'_'+str(i)+'.jpg'))
            i += 1

        siam_model.history['train_loss'].append(train_loss.result().numpy())
        siam_model.history['train_acc'].append(train_accuracy.result().numpy())
        siam_model.history['train_f1score'].append(train_f1score.result().numpy())

        siam_model.history['val_loss'].append(val_loss.result().numpy())
        siam_model.history['val_acc'].append(val_accuracy.result().numpy())
        siam_model.history['val_f1score'].append(val_f1score.result().numpy())

        if siam_model.history['val_f1score'][-1] >= best_f1score:
            last_improvement = 0
            siam_model.save_model()
            print("Model saved. f1score : {} --> {}".format(best_f1score, siam_model.history['val_f1score'][-1]))
            best_f1score = siam_model.history['val_f1score'][-1]
        else:
            last_improvement += 1

        if early_stopping is not None and last_improvement >= early_stopping:
            break

    plot_metrics(siam_model.history, epoch, plot_path)

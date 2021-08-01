import tensorflow as tf
import os

from src.model import Siamese
from src.metrics import precision, recall, accuracy, f1score
from src.utils import plot, plot_metrics, get_balance_factor, get_device
from src.dataset import get_dataset


def forward_step(model, inputs, device):
    with tf.device(device):
        output = model(inputs, training=False)
    return output


def forward_backward_step(model, inputs, label, optimizer, loss_fn, device):
    with tf.device(device):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = loss_fn(logits, label, activation=None, balance_factor=get_balance_factor(), training=True)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return logits, loss


def compute_metrics(logits, label, loss):
    precision_value = precision(logits, label),
    recall_value = recall(logits, label)
    f1score_value = f1score(precision_value, recall_value)
    accuracy_value = accuracy(logits, label)
    return [('loss', loss), ("f1", f1score_value), ("accuracy", accuracy_value)]


def plot_dataset_with_logits(model, dataset, save_path, epoch):
    for i, (image, template, labels) in zip(range(3), dataset.take(3)):
        predictions = model([image, template])
        filename = 'epoch_{}_sample_{}.jpg'.format(epoch+1, i)
        plot(image[0], template[0], labels[0], predictions[0],
             target='save', dest=os.path.join(save_path, filename))


def train(train_data_path, epochs, batch_size, plot_path, image_path,
          loss_fn, optimizer, early_stopping=None, plot_val_logits=True):

    device = get_device()
    print(f'Train on device {device}')
    siam_model = Siamese()
    training_set, validation_set, train_steps, val_steps = get_dataset(train_data_path, batch_size, show=False)

    best_loss = 0
    last_improvement = 0

    # Initialize dictionary to store the history
    siam_model.history = {'train_loss': [], 'val_loss': [], 'train_f1score': [],
                          'val_f1score': [], 'train_acc': [], 'val_acc': []}
    balance_factor = get_balance_factor()
    for epoch in range(epochs):

        print("Epoch: {}/{}".format(epoch + 1, epochs))

        train_loss = tf.metrics.Mean('train_loss')
        train_f1score = tf.metrics.Mean('train_f1')
        train_accuracy = tf.metrics.Mean('train_acc')

        train_progbar = tf.keras.utils.Progbar(train_steps)

        print("\nTRAIN")

        for b, (image, template, label) in enumerate(training_set):

            logits, loss = forward_backward_step(siam_model, [image, template], label, optimizer, loss_fn, device)

            metrics = compute_metrics(logits, label, loss)

            train_loss(loss)
            train_f1score(metrics[1][1])
            train_accuracy(metrics[2][1])

            train_progbar.update(b + 1, metrics)

        val_loss = tf.metrics.Mean('val_loss')
        val_f1score = tf.metrics.Mean('val_f1')
        val_accuracy = tf.metrics.Mean('val_acc')

        val_progbar = tf.keras.utils.Progbar(val_steps)

        # VALIDATION LOOP

        print("\nVALIDATE")

        for b, (image, template, label) in enumerate(validation_set):

            logits = forward_step(siam_model, [image, template], device)
            loss = loss_fn(logits, label, activation=None, balance_factor=balance_factor, training=False)

            metrics = compute_metrics(logits, label, loss)

            val_loss(loss)
            val_f1score(metrics[1][1])
            val_accuracy(metrics[2][1])

            val_progbar.update(b + 1, metrics)

        if plot_val_logits:
            plot_dataset_with_logits(siam_model, validation_set, image_path, epoch)

        siam_model.history['train_loss'].append(train_loss.result().numpy())
        siam_model.history['train_acc'].append(train_accuracy.result().numpy())
        siam_model.history['train_f1score'].append(train_f1score.result().numpy())

        siam_model.history['val_loss'].append(val_loss.result().numpy())
        siam_model.history['val_acc'].append(val_accuracy.result().numpy())
        siam_model.history['val_f1score'].append(val_f1score.result().numpy())

        if siam_model.history['val_loss'][-1] < best_loss:
            last_improvement = 0
            siam_model.save_model('checkpoint')
            target_loss = siam_model.history['val_loss'][-1]
            print(f'Model saved. validation loss : {best_loss} --> {target_loss}')
            best_loss = target_loss
        else:
            last_improvement += 1

        if early_stopping is not None and last_improvement >= early_stopping:
            break

    plot_metrics(siam_model.history, plot_path)

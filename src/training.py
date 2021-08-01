import tensorflow as tf
import os

from src.model import Siamese
from src.metrics import precision, recall, accuracy, f1score
from src.utils import plot, plot_metrics, get_balance_factor, get_device
from src.dataset import get_dataset


@tf.function
def forward_step(model, inputs, device):
    with tf.device(device):
        output = model(inputs, training=False)
    return output


@tf.function
def forward_backward_step(model, inputs, label, optimizer, loss_fn, device):
    with tf.device(device):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = loss_fn(logits, label, activation=None, balance_factor=get_balance_factor(), training=True)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return logits, loss


@tf.function
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

    training_set, validation_set, train_steps, val_steps = get_dataset(train_data_path, batch_size, show=False)

    train_metrics = {
        'train_loss': tf.metrics.Mean('train_loss'),
        'train_f1': tf.metrics.Mean('train_f1'),
        'train_acc': tf.metrics.Mean('train_acc'),
    }

    val_metrics = {
        'val_loss': tf.metrics.Mean('val_loss'),
        'val_f1': tf.metrics.Mean('val_f1'),
        'val_acc': tf.metrics.Mean('val_acc'),
    }
    model = Siamese()

    train_progbar = tf.keras.utils.Progbar(train_steps)
    val_progbar = tf.keras.utils.Progbar(val_steps)

    model = train_loop(model, training_set, validation_set, train_steps, val_steps, epochs, plot_path,
                       image_path, loss_fn, optimizer, train_metrics, val_metrics, train_progbar,
                       val_progbar, early_stopping, plot_val_logits)
    tf.print(model.history)


@tf.function
def train_loop(model, training_set, validation_set, train_steps, val_steps, epochs, plot_path,
               image_path, loss_fn, optimizer, train_metrics, val_metrics, train_progbar, val_progbar,
               early_stopping=None, plot_val_logits=True):

    device = get_device()
    print(f'Train on device {device}')

    best_loss = 1000000
    last_improvement = 0

    # Initialize dictionary to store the history
    model.history = {'train_loss': [], 'val_loss': [], 'train_f1score': [],
                     'val_f1score': [], 'train_acc': [], 'val_acc': []}
    balance_factor = get_balance_factor()

    train_loss = train_metrics['train_loss']
    train_f1score = train_metrics['train_f1']
    train_accuracy = train_metrics['train_acc']

    val_loss = val_metrics['val_loss']
    val_f1score = val_metrics['val_f1']
    val_accuracy = val_metrics['val_acc']

    for epoch in range(epochs):

        print(f'\nEpoch: {epoch+1}/{epochs}')

        train_loss.reset_states()
        train_f1score.reset_states()
        train_accuracy.reset_states()

        val_loss.reset_states()
        val_f1score.reset_states()
        val_accuracy.reset_states()

        print("\nTRAIN")
        for b, (image, template, label) in zip(range(train_steps), training_set.take(train_steps)):

            logits, loss = forward_backward_step(model, [image, template], label, optimizer, loss_fn, device)

            metrics = compute_metrics(logits, label, loss)

            train_loss(loss)
            train_f1score(metrics[1][1])
            train_accuracy(metrics[2][1])

            train_progbar.update(b+1)

        print("\nVALIDATE")
        for b, (image, template, label) in zip(range(val_steps), training_set.take(val_steps)):

            logits = forward_step(model, [image, template], device)
            loss = loss_fn(logits, label, activation=None, balance_factor=balance_factor, training=False)

            metrics = compute_metrics(logits, label, loss)

            val_loss(loss)
            val_f1score(metrics[1][1])
            val_accuracy(metrics[2][1])

            val_progbar.update(b+1)

        model.history['train_loss'].append(train_loss.result())
        model.history['train_acc'].append(train_accuracy.result())
        model.history['train_f1score'].append(train_f1score.result())

        model.history['val_loss'].append(val_loss.result())
        model.history['val_acc'].append(val_accuracy.result())
        model.history['val_f1score'].append(val_f1score.result())

        if tf.executing_eagerly():
            if plot_val_logits:
                plot_dataset_with_logits(model, validation_set, image_path, epoch)

            if model.history['val_loss'][-1] < best_loss:
                last_improvement = 0
                model.save_model('checkpoint')
                target_loss = model.history['val_loss'][-1]
                print(f'Model saved. validation loss : {best_loss} --> {target_loss}')
                best_loss = target_loss
            else:
                last_improvement += 1

            if early_stopping is not None and last_improvement >= early_stopping:
                break

    if tf.executing_eagerly():
        plot_metrics(model.history, plot_path)

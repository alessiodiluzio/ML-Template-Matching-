import os
import tensorflow as tf

from matplotlib import pyplot as plt
from src import X_MIN, X_MAX, Y_MIN, Y_MAX, IMAGE_DIM, CROP_SIZE


def get_filenames(path):
    if os.path.exists(path):
        abs_path = os.path.abspath(path)
        names = os.listdir(path)
        names = map(lambda name: os.path.join(abs_path, name), names)
        return list(names)
    return []


def get_device():
    device = 'cpu:0'
    if len(tf.config.list_physical_devices('GPU')) > 0:
        device = 'gpu:0'
    return device

def make_box_representation(boxes, outer_box_width):

    x_max = boxes[X_MAX]
    x_min = boxes[X_MIN]
    y_max = boxes[Y_MAX]
    y_min = boxes[Y_MIN]

    x, y = x_max - x_min, y_max - y_min

    inner_box = tf.ones((y, x))

    left_padding = tf.zeros((y, x_min))
    right_padding = tf.zeros((y, outer_box_width - x_max))

    ret = tf.concat([left_padding, inner_box, right_padding], axis=1)

    top_padding = tf.zeros((y_min, outer_box_width))
    bottom_padding = tf.zeros((outer_box_width - y_max, outer_box_width))

    ret = tf.concat([top_padding, ret, bottom_padding], axis=0)
    ret = tf.cast(ret, dtype=tf.uint8)
    return ret


def show_plot(image, template, label):
    fig = plt.figure()
    sub_plt = fig.add_subplot(1, 3, 1)
    sub_plt.set_title("Source")
    plt.imshow(image)
    sub_plt = fig.add_subplot(1, 3, 2)
    sub_plt.set_title("Template")
    plt.imshow(template)
    sub_plt = fig.add_subplot(1, 3, 3)
    sub_plt.set_title("Ground Truth")
    plt.imshow(label)
    plt.show()


def save_plot(image, template, label, dest):
    fig = plt.figure()
    sub_plt = fig.add_subplot(1, 3, 1)
    sub_plt.set_title("Source")
    plt.imshow(image)
    sub_plt = fig.add_subplot(1, 3, 2)
    sub_plt.set_title("Template")
    plt.imshow(template)
    sub_plt = fig.add_subplot(1, 3, 3)
    sub_plt.set_title("Ground Truth")
    plt.imshow(label)
    plt.savefig(dest)


def show_dataset_plot(dataset, samples):
    i = 0
    for image, template, label in dataset.take(samples):
        show_plot(image[i], template[i], label[i])
        i += 1


def save_dataset_plot(dataset, samples, dest):
    i = 0
    for image, template, label in dataset.take(samples):
        save_plot(image[i], template[i], label[i], os.path.join(dest, str(i)+'.jpg'))
        i += 1


def get_val_metric(metric_name, val_metrics):
    for m in val_metrics:
        if metric_name in m:
            return m


# TODO DA RIFARE
def plot_metrics(model_history, epochs, save_path):
    l_epochs = epochs
    epochs = range(epochs)
    train_metrics = []
    val_metrics = []
    for key in model_history:
        if 'train' in key:
            train_metrics.append(key)
        else:
            val_metrics.append(key)
    for metric in train_metrics:
        metric_name = metric.split('_')[1]
        mv = model_history[get_val_metric(metric_name, val_metrics)]
        mt = model_history[metric]
        labelt = 'Training'
        labelv = 'Validation'
        plt.figure()
        colorv = 'red'
        colort = 'blue'
        plt.plot(epochs, mt, color=colort, linestyle='-', label=labelt + ' ' + metric_name)
        plt.plot(epochs, mv, color=colorv, linestyle='-', label=labelv + ' ' + metric_name)
        plt.title(metric_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name + ' value')
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(save_path, metric_name + '_' + str(l_epochs) + '.jpg'))
        plt.pause(0.001)
        plt.close()


def get_balance_factor():
    return (CROP_SIZE * CROP_SIZE)/(IMAGE_DIM * IMAGE_DIM)

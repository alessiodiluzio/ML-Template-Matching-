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


def create_label_mask(label_mask):
    label_mask = tf.argmax(label_mask, axis=-1)
    label_mask = label_mask[..., tf.newaxis]
    return label_mask


def save_plot(image, template, label=None, logit=None, dest='.'):
    n_plot = 2
    if label is not None:
        n_plot += 1
    if logit is not None:
        n_plot += 1
    fig = plt.figure()
    sub_plt = fig.add_subplot(1, n_plot, 1)
    sub_plt.set_title("Source")
    plt.imshow(image)
    sub_plt = fig.add_subplot(1, n_plot, 2)
    sub_plt.set_title("Template")
    plt.imshow(template)
    if label is not None:
        sub_plt = fig.add_subplot(1, n_plot, 3)
        sub_plt.set_title("Ground Truth")
        plt.imshow(label)
    if logit is not None:
        logit = tf.squeeze(create_label_mask(logit), axis=-1)
        sub_plt = fig.add_subplot(1, n_plot, 4)
        sub_plt.set_title("Prediction")
        plt.imshow(logit)
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


def plot_metrics(model_history, save_path):
    metric_names = [key.split('_')[1] for key in model_history if 'train' in key]
    for metric_name in metric_names:
        mv = model_history['val_'+metric_name]
        mt = model_history['train_'+metric_name]
        label_t = 'Training'
        label_v = 'Validation'
        plt.figure()
        color_v = 'red'
        color_t = 'blue'
        plt.plot(range(len(mt)), mt, color=color_t, linestyle='-', label=label_t + ' ' + metric_name)
        plt.plot(range(len(mv)), mv, color=color_v, linestyle='-', label=label_v + ' ' + metric_name)
        plt.title(metric_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name + ' value')
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(save_path, metric_name + '_' + str(len(mv)) + '.jpg'))
        plt.pause(0.001)
        plt.close()


def get_balance_factor():
    return (CROP_SIZE * CROP_SIZE)/(IMAGE_DIM * IMAGE_DIM)

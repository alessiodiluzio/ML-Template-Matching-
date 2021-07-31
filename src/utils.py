import os
import tensorflow as tf

from matplotlib import pyplot as plt
from src import X_1, X_2, Y_1, Y_2, IMAGE_DIM, CROP_SIZE


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


def make_prediction(boxes, x_scale_factor=CROP_SIZE, y_scale_factor=CROP_SIZE, outer_box_dim=IMAGE_DIM):

    x1 = boxes[X_1]
    y1 = boxes[Y_1]

    x2 = boxes[X_2]
    y2 = boxes[Y_2]

    x1 = tf.cast(x1 * x_scale_factor + x_scale_factor / 2, dtype=tf.int32)
    y1 = tf.cast(y1 * y_scale_factor + y_scale_factor / 2, dtype=tf.int32)
    x2 = tf.cast(x2 * x_scale_factor + x_scale_factor / 2, dtype=tf.int32)
    y2 = tf.cast(y2 * y_scale_factor + y_scale_factor / 2, dtype=tf.int32)

    x1_tmp = tf.minimum(x1, x2)
    x2 = tf.maximum(x1, x2)
    x1 = x1_tmp

    y1_tmp = tf.maximum(y1, y2)
    y2 = tf.maximum(y1, y2)
    y1 = y1_tmp

    boxes = tf.stack([x1, y1, x2, y2], axis=0)
    prediction = make_label(boxes, outer_box_dim)
    return prediction


def make_label(boxes, outer_box_dim, scale=False):

    x1 = boxes[X_1]
    y1 = boxes[Y_1]

    x2 = boxes[X_2]
    y2 = boxes[Y_2]

    x, y = x2 - x1, y2 - y1

    #x = tf.maximum(0, x)
    #y = tf.maximum(0, y)

    inner_box = tf.ones((y, x))
    paddings = [[y1, outer_box_dim - y2], [x1, outer_box_dim - x2]]
    ret = tf.pad(inner_box, paddings, mode='CONSTANT')

    ret = tf.expand_dims(ret, axis=-1)
    ret = tf.cast(ret, dtype=tf.float32)
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
        logit = tf.squeeze(logit, axis=-1)
        sub_plt = fig.add_subplot(1, n_plot, 4)
        sub_plt.set_title("Prediction")
        plt.imshow(logit)
    plt.savefig(dest)
    plt.pause(0.001)
    plt.close()


def show_dataset_plot(dataset, samples):
    i = 0
    for image, template, label in dataset.take(samples):
        show_plot(image[i], template[i], label[i])
        i += 1


def save_dataset_plot(dataset, samples, dest):
    i = 0
    for image, template, label in dataset.take(samples):
        print(label[i].shape)
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

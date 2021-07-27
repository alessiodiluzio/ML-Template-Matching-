""" Implementation of Tensorflow input pipeline for Template Matching"""

import tensorflow as tf
from src import CHANNELS, IMAGE_DIM, DATA_PATH, BATCH_SIZE, X_MIN, Y_MIN, CROP_SIZE, OUTPUT_DIM
from src.utils import make_box_representation, get_filenames, show_dataset_plot


@tf.function
def load_image(filename):
    """
    1)
    Load image from filesystem
    :param filename: path to image
    :return: Tensorflow decoded image.
    """
    raw = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(raw, channels=CHANNELS)
    return image


@tf.function
def preprocess(image):
    """
    2)
    Resize and normalize an image, build a random box that will be used in the following crop step
    :param image: Tensorflow decoded image
    :return: image and the coordinates of a box specified by [x_min, x_max, y_min, y_max].
    """
    image = tf.image.resize(image, [IMAGE_DIM, IMAGE_DIM])
    image /= IMAGE_DIM
    x1 = tf.random.uniform(shape=[1], minval=0, maxval=IMAGE_DIM - CROP_SIZE, dtype=tf.int32)
    y1 = tf.random.uniform(shape=[1], minval=0, maxval=IMAGE_DIM - CROP_SIZE, dtype=tf.int32)
    x1 = tf.cast(x1, dtype=tf.float32)
    y1 = tf.cast(y1, dtype=tf.float32)
    y2 = tf.math.add(y1, CROP_SIZE)
    x2 = tf.math.add(x1, CROP_SIZE)
    boxes = tf.concat([y1, x1, y2, x2], axis=0)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    return image, boxes


@tf.function
def extract_crop(image, boxes):
    """
    3)
    Extract crop from an image at specified coordinates.
    :param image: Tensorflow decoded image
    :param boxes: the coordinates of a box specified by [x_min, x_max, y_min, y_max].
    :return: Source image, the crop extracted from source image, the coordinates of the box.
    """
    # box_indices = [NUM_BOXES-1]
    tmp_boxes = boxes/255
    # tmp_boxes = tf.expand_dims(tmp_boxes, axis=0)
    # tmp_img = tf.expand_dims(image, axis=0)
    # template = tf.image.crop_and_resize(tmp_img, tmp_boxes, box_indices, CROP_BOX)
    """
    offset_width = tf.cast(boxes[X_MIN]-1, dtype=tf.int32)
    offset_width = tf.cond(tf.less(offset_width, 0), lambda: tf.add(offset_width, 1), lambda: offset_width)

    offset_height = tf.cast(boxes[Y_MIN]-1, dtype=tf.int32)
    offset_height = tf.cond(tf.less(offset_height, 0), lambda: tf.add(offset_height, 1), lambda: offset_height)

    template = tf.image.pad_to_bounding_box(template[0], offset_height, offset_width, IMAGE_DIM, IMAGE_DIM)
    """
    begin = tf.stack([tf.cast(boxes[X_MIN], dtype=tf.int32), tf.cast(boxes[Y_MIN], dtype=tf.int32), 0], axis=0)
    template = tf.slice(image, begin, size=[CROP_SIZE, CROP_SIZE, 3])
    # template = tf.image.resize(template[0], [IMAGE_DIM, IMAGE_DIM])
    return image, template, boxes


@tf.function
def generate_ground_truth(image, template, boxes):
    """
    4)
    Generate a ground truth label for the crop extracted from the source image
    :param image: Tensorflow decoded image
    :param template: A crop extracted from source image at coordinates specified by boxes param
    :param boxes: the coordinates of a box specified by [x_min, x_max, y_min, y_max].
    :return: Source image, crop, a 1D image that has the same size of source image and it is composed
                by 'ones' in the pixel that match with crop coordinates, the remaining pixels have value zero
    """
    tmp_boxes = boxes * (OUTPUT_DIM/IMAGE_DIM)
    tmp_boxes = tf.cast(tmp_boxes, dtype=tf.int32)
    label = make_box_representation(tmp_boxes, OUTPUT_DIM)
    # boxes = tf.stack([boxes[X_MIN], boxes[Y_MIN]], axis=0)
    return image, template, label


# TODO
@tf.function
def perturb(image, template, label):
    """
    5) (optional)
    Randomly perturb source image and Ground truth label
    :param image: Tensorflow decoded image
    :param template: A crop extracted from source image at coordinates specified by boxes param
    :param label: Ground truth label for crop against source image.
    :return: Source image randomly perturbed, the crop, the ground truth perturbed in the same way of source image
    """
    return image, template, label


def make_dataset(images_path, batch_size, augmentation=False):
    """
    Build a dataset from a list of image
    :param images_path: List containing path of images
    :param batch_size: Hyperparam for net training, the dataset is processed in batch of dim batch_size
    :param augmentation: if true, images are randomly perturbed.
    :return: A dataset ready for a training.
    """
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(len(images_path))
    dataset = dataset.map(load_image)  # 1)
    dataset = dataset.map(preprocess)  # 2)
    dataset = dataset.map(extract_crop)  # 3)
    dataset = dataset.map(generate_ground_truth)  # 4)
    if augmentation:
        dataset = dataset.map(perturb)  # 5)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def get_dataset(data_path=DATA_PATH, batch_size=BATCH_SIZE, split_perc=0.7, show=False):
    """
    Build training and validation set ready for training phase.
    :param data_path: Path to the folder that contains the dataset images
    :param batch_size: Hyperparam for net training, the dataset is divided in batch of dim batch_size
    :param split_perc: The percentual of dataset that is used for training (the remaining is used as validation set)
    :param show: whether show samples from training set.
    :return: datasets ready for training and validation, training step and validation step
    """
    images = get_filenames(data_path)
    val_index = int(len(images) * split_perc)
    training_images = images[:val_index]
    validation_images = images[val_index:]
    training_set = make_dataset(training_images, batch_size, augmentation=True)
    validation_set = make_dataset(validation_images, batch_size)
    training_step = int(len(training_images)/BATCH_SIZE)  # training step = | TRAINING_SET |/batch_size
    validation_step = int(len(validation_images)/BATCH_SIZE)  # validation step = | VALIDATION_SET |/batch_size
    if show:
        show_dataset_plot(training_set, 3)
        # save_dataset_plot(training_set, 3, 'checkpoint')
    return training_set, validation_set, training_step, validation_step

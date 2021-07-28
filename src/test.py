import tensorflow as tf
import os

from src.model import Siamese
from src.utils import save_plot, get_device


def test(test_set, output_path):

    device = get_device()
    siam_model = Siamese(checkpoint_dir="checkpoint", device=device)
    for b, (image, template) in enumerate(test_set):
        predictions = tf.nn.sigmoid(siam_model.forward([image, template]))
        for im in range(predictions[0]):
            save_plot(image, template, dest=os.path.join(output_path, str(im)+'.jpg'))

import tensorflow as tf
import os
from src.layers import SiameseConv2D, CorrelationFilter
from src.loss import compute_cross_entropy_loss
from src.utils import get_balance_factor


class Siamese(tf.keras.Model):

    def __init__(self, checkpoint_dir, device):
        super(Siamese, self).__init__(name='Siamese')
        self._checkpoint_dir = os.path.join(checkpoint_dir, self.name)
        self._device = device
        self._alexnet_encoder = AlexnetEncoder()
        self._balance_factor = get_balance_factor()
        self._correlation_filter = CorrelationFilter()

    def call(self, input_tensor, training=False, **kwargs):
        x, z = self._alexnet_encoder(input_tensor, training)
        net_final = self._correlation_filter([x, z])
        return net_final

    def build(self, input_shape):
        super(Siamese, self).build(input_shape)
        self._alexnet_encoder.build(input_shape)

    @tf.function
    def forward(self, *args, **kwargs):
        with tf.device(self._device):
            output = self.call(*args, **kwargs)
        return output

    @tf.function
    def forward_backward_pass(self, inputs, label, optimizer):
        with tf.device(self._device):
            with tf.GradientTape() as tape:
                logits = self.call(inputs, training=True)
                loss = compute_cross_entropy_loss(logits, label, self._balance_factor, training=True)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return logits, loss

    def get_config(self):
        return {"checkpoint_dir": self._checkpoint_dir, "device": self._device}

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)

    def save_model(self):
        pass
        # self.save(self._checkpoint_dir)


class AlexnetEncoder(tf.keras.Model):

    def __init__(self):
        super(AlexnetEncoder, self).__init__(name='alexnet_encoder')
        self.conv1 = SiameseConv2D(filters=96, kernel_size=(11, 11), strides=2,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_1')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, name='Max_Pool_1')

        self.conv2 = SiameseConv2D(filters=256, kernel_size=(5, 5), strides=1,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_2')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, name='Max_Pool_2')

        self.conv3 = SiameseConv2D(filters=192, kernel_size=(3, 3), strides=1,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_3')
        self.conv4 = SiameseConv2D(filters=192, kernel_size=(3, 3), strides=1,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_4')
        self.conv5 = SiameseConv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid',
                                   activation=None, name='Conv_5')

    def call(self, input_tensor, training=False, **kwargs):
        output = self.conv1(input_tensor, training)

        x = self.pool1(output[0])
        z = self.pool1(output[1])

        x, z = self.conv2([x, z], training)

        x = self.pool2(x)
        z = self.pool2(z)

        x, z = self.conv3([x, z], training)
        x, z = self.conv4([x, z], training)
        x, z = self.conv5([x, z], training)
        return x, z

"""This module defines a class for a Twin-Model for MNIST data."""
from enum import Enum, auto
import tensorflow as tf


class Subtask(Enum):
    """
    Enumerator for the different subtasts:
    LARGER_FIVE: solves a+b >= 5
    DIFFERNECE: solves a-b = y.
    """

    LARGER_FIVE = auto()
    DIFFERENCE = auto()


class TwinModelMnist(tf.keras.Model):
    def __init__(self, subtask: Subtask, optimizer=tf.keras.optimizers.Adam()):
        super().__init__()
        self.metrics_list = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Mean(name="loss"),
        ]
        self.optimizer = optimizer
        self._units_in_layer = 128
        if subtask == Subtask.DIFFERENCE:
            self.loss_function = tf.keras.losses.CategoricalCrossentropy()
            self._units_in_output = 19
            _activation = tf.nn.relu
            _activation_output = tf.nn.softmax
        if subtask == Subtask.LARGER_FIVE:
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
            self._units_in_output = 1
            _activation = tf.nn.relu
            _activation_output = tf.nn.sigmoid

        self.dense1 = tf.keras.layers.Dense(
            self._units_in_layer, activation=_activation
        )
        self.dense2 = tf.keras.layers.Dense(
            self._units_in_layer, activation=_activation
        )
        self.out_layer = tf.keras.layers.Dense(
            self._units_in_output, activation=_activation_output
        )

    def call(self, images, training=False):
        img1, img2 = images

        img1_x = self.dense1(img1)
        img1_x = self.dense2(img1_x)

        img2_x = self.dense1(img2)
        img2_x = self.dense2(img2_x)

        combined_x = tf.concat([img1_x, img2_x], axis=1)
        return self.out_layer(combined_x)

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    @tf.function
    def _step(self, data, training=False):
        image_1, image_2, label = data
        if training:
            with tf.GradientTape() as tape:
                output = self((image_1, image_2), training=training)
                loss = self.loss_function(label, output)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            output = self((image_1, image_2), training=False)
            loss = self.loss_function(label, output)
        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def train_step(self, data):
        return self._step(data, training=True)

    @tf.function
    def test_step(self, data):
        return self._step(data)

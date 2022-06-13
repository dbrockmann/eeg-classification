
import tensorflow as tf
import numpy as np

class Decoder(tf.keras.Model):

    def __init__(self, input_dim):
        """
        Initializes the Decoder model layers

        Args:
            input_dim: input dimensions
        """

        super(Decoder, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(128, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(np.prod(input_dim), activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape(input_dim)

    @tf.function
    def call(self, x, training):
        """
        Compute output of the model given an input

        Args:
            x: the input
            training: flag stating if in training mode

        Returns:
            output of the model
        """

        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.reshape(x, training=training)
        return x


import tensorflow as tf

class Decoder(tf.keras.Model):

    def __init__(self, input_dim):
        """
        Initializes the Decoder model layers

        Args:
            input_dim: input dimensions
        """

        super(Decoder, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(input_dim, activation='sigmoid')

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
        return x

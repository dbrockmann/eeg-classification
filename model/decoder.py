
import tensorflow as tf

class Decoder(tf.keras.Model):

    def __init__(self):
        """
        Initializes the Decoder model layers
        """

        super(Decoder, self).__init__()

        self.dense = tf.keras.layers.Dense(64)

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

        x = self.dense(x, training=training)
        return x

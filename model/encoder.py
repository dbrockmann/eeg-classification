
import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim):
        """
        Initializes the Encoder model layers

        Args:
            latent_dim: dimensions of the latent space
        """

        super(Encoder, self).__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(latent_dim, activation='relu')

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

        x = self.flatten(x, training=training)
        x = self.dense(x, training=training)
        return x


import tensorflow as tf

from encoder import Encoder
from decoder import Decoder

class Autoencoder(tf.keras.Model):

    def __init__(self, latent_dim):
        """
        Initializes the Autoencoder consisting of an Encoder and a Decoder

        Args:
            latent_dim: dimensions of the latent space
        """

        super(Autoencoder, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    @tf.function
    def call(self, x, training):
        """
        Compute output of the model given an input

        Args:
            x: the input
            training: flag stating if in training mode
        """

        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return x

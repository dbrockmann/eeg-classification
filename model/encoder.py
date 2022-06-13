
import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim):
        """
        Initializes the Encoder model layers

        Args:
            latent_dim: dimensions of the latent space
        """

        super(Encoder, self).__init__()

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(latent_dim, activation='sigmoid')
        self.regularization = tf.keras.layers.ActivityRegularization(l1=1e-3)

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
        x = self.regularization(x, training=training)
        return x


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric


class Sampling(Layer):
    """
    Sampling layer that uses mean and log_var for sampling
    """

    def call(self, x):
        """
        Compute output by sampling from input

        Args:
            x: mean and log_var

        Returns:
            sampled output
        """

        mean, log_var = tf.split(x, num_or_size_splits=2, axis=1)
        epsilon = tf.random.normal(shape=mean.shape)
        return mean + epsilon * tf.exp(0.5 * log_var)


class VariationalAutoencoder(Model):
    """
    Variational autoencoder
    """

    def __init__(self, input_dim, latent_dim):
        """
        Initialize layers

        Args:
            input_dim: input dimensions
            latent_dim: bottleneck dimensions, latent space

        Returns:
            variational autoencoder model, encoder model
        """

        super(VariationalAutoencoder, self).__init__()

        self.encoder = Sequential([
            Dense(
                units=128, 
                activation='elu',
                kernel_initializer='he_normal'
            ),
            Dense(
                units=64, 
                activation='elu',
                kernel_initializer='he_normal'
            ),
            Dense(
                units=2*latent_dim
            )
        ])
        self.mean = None
        self.log_var = None

        self.sampling = Sampling()

        self.decoder = Sequential([
            Dense(
                units=64, 
                activation='elu',
                kernel_initializer='he_normal'
            ),
            Dense(
                units=128, 
                activation='elu',
                kernel_initializer='he_normal'
            ),
            Dense(
                units=input_dim,
                activation='sigmoid'
            )
        ])

        self.reconstruction_loss = MeanSquaredError()

    def call(self, x, training):
        """
        Compute output of the model given an input

        Args:
            x: the input
            training: flag stating if in training mode

        Returns:
            output of the model
        """

        x = self.encoder(x, training=training)
        self.mean, self.log_var = tf.split(x, num_or_size_splits=2, axis=1)
        x = self.sampling(x, training=training)
        x = self.decoder(x, training=training)
        return x

    def loss(self, y, y_pred):
        """
        Compute loss

        Args:
            y: target
            y_pred: prediction

        Returns:
            total loss
        """

        reconstruction_loss = self.reconstruction_loss(y, y_pred)

        kl_loss = -0.5 * (1 + self.log_var - tf.square(self.mean) - tf.exp(self.log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        loss = reconstruction_loss + kl_loss

        return loss


def build_variational_ae(input_dim, latent_dim):
    """
    Build variational autoencoder

    Args:
        input_dim: input dimensions
        latent_dim: bottleneck dimensions, latent space
    """

    variational_ae = VariationalAutoencoder(input_dim, latent_dim)

    encoder = Sequential([
        variational_ae.encoder,
        variational_ae.sampling
    ])

    variational_ae.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss=variational_ae.loss,
        metrics=[
            MeanSquaredErrorMetric(name='MSE')
        ]
    )

    return variational_ae, encoder

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

        self.dense = tf.keras.layers.Dense(
            np.prod(input_dim) // 8 * 64
        )

        self.reshape1 = tf.keras.layers.Reshape(
            (np.prod(input_dim) // 8, 64)
        )

        self.conv1 = tf.keras.layers.Conv1DTranspose(
            64, kernel_size=3, padding='same', activation='relu'
        )

        self.upsampling1 = tf.keras.layers.UpSampling1D(
            size=2
        )

        self.conv2 = tf.keras.layers.Conv1DTranspose(
            32, kernel_size=3, padding='same', activation='relu'
        )

        self.upsampling2 = tf.keras.layers.UpSampling1D(
            size=2
        )

        self.conv3 = tf.keras.layers.Conv1DTranspose(
            16, kernel_size=3, padding='same', activation='relu'
        )

        self.upsampling3 = tf.keras.layers.UpSampling1D(
            size=2
        )

        self.conv4 = tf.keras.layers.Conv1D(
            1, kernel_size=3, padding='same', activation='sigmoid'
        )

        self.reshape2 = tf.keras.layers.Reshape(
            input_dim
        )

        #self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        #self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        #self.dense3 = tf.keras.layers.Dense(np.prod(input_dim), activation='sigmoid')
        #self.reshape = tf.keras.layers.Reshape(input_dim)

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
        x = self.reshape1(x, training=training)

        x = self.conv1(x, training=training)
        x = self.upsampling1(x, training=training)

        x = self.conv2(x, training=training)
        x = self.upsampling2(x, training=training)

        x = self.conv3(x, training=training)
        x = self.upsampling3(x, training=training)
        
        x = self.conv4(x, training=training)
        x = self.reshape2(x, training=training)

        #x = self.dense1(x, training=training)
        #x = self.dense2(x, training=training)
        #x = self.dense3(x, training=training)
        #x = self.reshape(x, training=training)
        return x

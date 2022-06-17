
import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim):
        """
        Initializes the Encoder model layers

        Args:
            latent_dim: dimensions of the latent space
        """

        super(Encoder, self).__init__()


        self.reshape = tf.keras.layers.Reshape(
            (-1, 1)
        )

        self.conv1 = tf.keras.layers.Conv1D(
            16, kernel_size=3, padding='same', activation='relu'
        )
        
        self.pool1 = tf.keras.layers.MaxPooling1D(
            pool_size=2
        )

        self.conv2 = tf.keras.layers.Conv1D(
            32, kernel_size=3, padding='same', activation='relu'
        )

        self.pool2 = tf.keras.layers.MaxPooling1D(
            pool_size=2
        )

        self.conv3 = tf.keras.layers.Conv1D(
            64, kernel_size=3, padding='same', activation='relu'
        )

        self.pool3 = tf.keras.layers.MaxPooling1D(
            pool_size=2
        )

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            latent_dim, activation='relu'
        )

        #self.flatten = tf.keras.layers.Flatten()
        #self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        #self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        #self.dense3 = tf.keras.layers.Dense(latent_dim, activation='sigmoid')
        #self.regularization = tf.keras.layers.ActivityRegularization(l1=1e-3)


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

        x = self.reshape(x, training=training)
        x = self.conv1(x, training=training)
        x = self.pool1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.pool2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool3(x, training=training)
        x = self.flatten(x, training=training)
        x = self.dense(x, training=training)

        #x = self.flatten(x, training=training)
        #x = self.dense1(x, training=training)
        #x = self.dense2(x, training=training)
        #x = self.dense3(x, training=training)
        #x = self.regularization(x, training=training)
        return x

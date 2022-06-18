
import tensorflow as tf

class Classifier(tf.keras.Model):

    def __init__(self, class_dim):
        """
        Initializes the Classifier model layers

        Args:
            class_dim: number of classes
        """

        super(Classifier, self).__init__()

        self.dense1 = tf.keras.layers.Dense(
            32, activation='relu'
        )

        self.dense2 = tf.keras.layers.Dense(
            16, activation='relu'
        )

        self.dense3 = tf.keras.layers.Dense(
            class_dim, activation='softmax'
        )

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

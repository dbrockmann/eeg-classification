
import tensorflow as tf

class Classifier(tf.keras.Model):

    def __init__(self, class_dim):
        """
        Initializes the Classifier model layers

        Args:
            class_dim: number of classes
        """

        super(Classifier, self).__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.softmax = tf.keras.layers.Dense(class_dim, activation='softmax')

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
        x = self.softmax(x)
        return x

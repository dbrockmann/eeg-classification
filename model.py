
from tensorflow.keras import Model
from tensorflow import function as tf_function

class Model_From_Layers(Model):

    def __init__(self, layers):
        """
        Initializes the model layers

        Args:
            layers: layers of the model
        """

        super(Model_From_Layers, self).__init__()

        self.layers = layers

    @tf_function
    def call(self, x, training):
        """
        Compute output of the model by feeding
        the input through every layer

        Args:
            x: the input
            training: flag stating if in training mode

        Returns:
            output of the model
        """

        for layer in self.layers:
            x = layer(x, training=training)
        
        return x

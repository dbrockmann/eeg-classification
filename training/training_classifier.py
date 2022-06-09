
import tensorflow as tf

from training import train_model


def train_classifier(model, train_data, test_data):
    """
    Train classifier model

    Args:
        model: classifier model
        train_data: train dataset
        test_data: test dataset
    """

    # number of training epochs
    epochs = 10

    # learning rate for optimizer
    learning_rate = 0.001

    # loss function: categorical cross entropy
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # optimizer: Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # train model
    train_loss, test_loss = train_model(model, train_data, test_data, loss_fn, optimizer, epochs)

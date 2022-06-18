
import tensorflow as tf

from training.training import train_model


def train_classifier(model, train_ds, test_ds, show=False):
    """
    Train classifier model

    Args:
        model: classifier model
        train_ds: train dataset
        test_ds: test dataset
        show: print loss after every epoch

    Returns:
        aggregated training and test losses
    """

    # number of training epochs
    epochs = 10

    # learning rate for optimizer
    learning_rate = 0.001

    # loss function: categorical cross entropy
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # optimizer: Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # train model
    train_loss, test_loss = train_model(model, train_ds, test_ds, loss_fn, optimizer, epochs, show)

    return train_loss, test_loss

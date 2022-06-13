
import tensorflow as tf

from training import train_model


def train_autoencoder(model, train_data, test_data):
    """
    Train autoencoder model

    Args:
        model: autoencoder model
        train_data: train dataset
        test_data: test dataset
    """

    # cache, batch and prefetch
    train_data = train_data.cache().batch(32).prefetch(8)
    test_data = test_data.cache().batch(32).prefetch(8)

    # number of training epochs
    epochs = 10

    # learning rate for optimizer
    learning_rate = 0.05

    # loss function: mean squared error
    loss_fn = tf.keras.losses.MeanSquaredError()

    # optimizer: Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # train model
    train_loss, test_loss = train_model(model, train_data, test_data, loss_fn, optimizer, epochs)


import tensorflow as tf

from training.training import train_model


def train_autoencoder(model, train_ds, test_ds, show=False):
    """
    Train autoencoder model

    Args:
        model: autoencoder model
        train_ds: train dataset
        test_ds: test dataset

    Returns:
        aggregated training and test losses
    """

    # cache, batch and prefetch
    train_ds = train_ds.cache().batch(12).prefetch(8)
    test_ds = test_ds.cache().batch(12).prefetch(8)

    # number of training epochs
    epochs = 10

    # learning rate for optimizer
    learning_rate = 0.05

    # loss function: MSE
    loss_fn = tf.keras.losses.MeanSquaredError()
    #loss_fn = tf.keras.losses.BinaryCrossentropy()

    # optimizer: Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # train model
    train_loss, test_loss = train_model(model, train_ds, test_ds, loss_fn, optimizer, epochs, show)

    return train_loss, test_loss



import tensorflow as tf
import numpy as np


def train_model(model, train_ds, test_ds, loss_fn, optimizer, epochs):
    """
    Training loop with testing

    Args:
        model: the model to train
        train_ds: train dataset
        test_ds: test dataset
        loss_fn: loss function
        optimizer: the optimizer
        epochs: number of epochs to train

    Returns:
        aggregated training and test losses
    """

    # train loss aggregator
    train_loss_agg = []

    # test loss aggregator
    test_loss_agg = []

    # test on train data before first training
    train_loss = test(model, train_ds, loss_fn)
    train_loss_agg.append(train_loss)

    # test on test data before first training
    test_loss = test(model, test_ds, loss_fn)
    test_loss_agg.append(test_loss)

    # repeat training/testing for number of epochs
    for _ in range(epochs):

        # training
        train_loss = train(model, train_ds, loss_fn, optimizer)
        train_loss_agg.append(train_loss)

        # testing
        test_loss = test(model, test_ds, loss_fn)
        test_loss_agg.append(test_loss)

    # return aggregated training and test losses
    return train_loss_agg, test_loss_agg


def train(model, ds, loss_fn, optimizer, training=True):
    """
    Performs training on a full dataset and aggregates the losses

    Args:
        model: the model to train
        ds: the train dataset
        loss_fn: loss function
        optimizer: the optimizer
        training: flag stating if in training mode

    Returns:
        mean loss
    """

    # loss aggregator
    loss_agg = []

    # for input and target in dataset
    for x, t in ds:

        # calculate loss
        loss = train_step(model, x, t, loss_fn, optimizer, training)

        # append loss
        loss_agg.append(loss.numpy())

    # calculate mean of losses
    loss = np.mean(loss_agg)

    # reaturn mean loss
    return loss


def train_step(model, x, t, loss_fn, optimizer, training):
    """
    Peforms a training step by doing a forward and backward pass
    through the network

    Args:
        model: the model to train
        x: input
        t: target output
        loss_fn: loss function
        optimizer: the optimizer
        training: flag stating if in training mode

    Returns:
        calculated loss
    """

    with tf.GradientTape() as tape:

        # forward step
        prediction = model(x, training)

        # calculate loss
        loss = loss_fn(t, prediction)

        # calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)

    # update weights by applying the gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # return calculated loss
    return loss


def test(model, ds, loss_fn, training=False):
    """
    Tests performance on a full dataset and aggregates the losses

    Args:
        model: the model to train
        ds: the train dataset
        loss_fn: loss function
        training: flag stating if in training mode

    Returns:
        mean loss
    """

    # loss aggregator
    loss_agg = []

    # for input and target in dataset
    for x, t in ds:

        # calculate loss
        loss = test_step(model, x, t, loss_fn, training)

        # append loss
        loss_agg.append(loss.numpy())

    # calculate mean of losses
    loss = np.mean(loss_agg)

    # reaturn mean loss
    return loss


def test_step(model, x, t, loss_fn, training):
    """
    Calculates loss by performing a forward
    step on a test datapoint

    Args:
        model: the model to test
        x: input
        t: target output
        loss_fn: loss function
        training: flag stating if in training mode

    Returns:
        calculated loss
    """

    # forward step
    prediction = model(x, training)

    # calculate loss
    loss = loss_fn(t, prediction)

    # reaturn loss
    return loss

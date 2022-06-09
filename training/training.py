
import tensorflow as tf
import numpy as np


def train(model, data, loss_fn, optimizer, training=True):
    """
    Performs training on a full dataset and aggregates the losses

    Args:
        model: the model to train
        data: the train dataset
        loss_fn: loss function
        optimizer: the optimizer
        training: flag stating if in training mode

    Returns:
        mean loss
    """

    # loss aggregator
    loss_agg = []

    # for input and target in dataset
    for x, t in data:

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

    with tf.GradientType() as type:

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


def test(model, data, loss_fn, training=False):
    """
    Tests performance on a full dataset and aggregates the losses

    Args:
        model: the model to train
        data: the train dataset
        loss_fn: loss function
        training: flag stating if in training mode

    Returns:
        mean loss
    """

    # loss aggregator
    loss_agg = []

    # for input and target in dataset
    for x, t in data:

        # calculate loss
        loss = test_step(model, x, t, loss_fn, training)

        # append loss
        loss_agg.append(loss.numpy())

    # calculate mean of losses
    loss = np.mean(loss_agg)

    # reaturn mean loss
    return loss


def test_step(model, data, loss_fn, training=False):
    """
    Calculates loss by performing a forward
    step on a test datapoint

    Args:
        model: the model to test
        data: the test dataset
        loss_fn: loss function
        training: flag stating if in training mode

    Returns:
        calculated loss
    """

    # forward step
    prediction = model(x, testing)

    # calculate loss
    loss = loss_fn(t, prediction)

    # reaturn loss
    return loss

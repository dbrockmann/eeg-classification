
from tensorflow import GradientTape
import numpy as np


def train_from_config(model, train_ds, test_ds, config, show=False):
    """
    Training loop with testing from configuration

    Args:
        model: the model to train
        train_ds: train dataset
        test_ds: test dataset
        config: dictionary with training configurations
        show: print loss after every epoch

    Returns:
        aggregated training and test losses
    """

    # batch and prefetch
    train_ds = train_ds.batch(config['batch_size']).prefetch(1)
    test_ds = test_ds.batch(config['batch_size']).prefetch(1)

    # train loss aggregator
    train_loss_agg = []

    # test loss aggregator
    test_loss_agg = []

    # test on train data before first training
    train_loss = test(model, train_ds, config['loss_function'])
    train_loss_agg.append(train_loss)

    # test on test data before first training
    test_loss = test(model, test_ds, config['loss_function'])
    test_loss_agg.append(test_loss)

    # print loss if show flag is set
    if (show):
        print(f'Epoch 0: train loss {train_loss}, test loss {test_loss}')

    # repeat training/testing for number of epochs
    for epoch in range(config['epochs']):

        # training
        train_loss = train(model, train_ds, config['loss_function'], config['optimizer'])
        train_loss_agg.append(train_loss)

        # testing
        test_loss = test(model, test_ds, config['loss_function'])
        test_loss_agg.append(test_loss)

        # print loss if show flag is set
        if (show):
            print(f'Epoch {epoch + 1}: train loss {train_loss}, test loss {test_loss}')

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

    with GradientTape() as tape:

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


from tensorflow import GradientTape
import numpy as np


def train_model(model, train_ds, test_ds, epochs=20, show=True):
    """
    Training loop with testing

    Args:
        model: the model to train
        train_ds: train dataset
        test_ds: test dataset
        epochs: epochs to train
        show: print loss after every epoch

    Returns:
        aggregated training and test losses
    """

    # train loss aggregator
    train_loss_agg = []

    # test metrics aggregator
    metrics_agg = []

    # test on train data before first training
    train_loss = train(model, train_ds, training=False)
    train_loss_agg.append(train_loss)

    # test on test data before first training
    test_metric = test(model, test_ds)
    metrics_agg.append(test_metric)

    # print loss and metrics to console
    def print_status(epoch, loss, metrics):
        print(f'{epoch:5}: {loss:10.6}',
              f'{", ".join(["{0}: {1:10.6}".format(name, value) for name, value in metrics.items()])}',
              sep=5*' ')

    # print model summary and loss if show flag is set
    if (show):
        model.summary()
        print_status(0, train_loss, test_metric)

    # repeat training/testing for number of epochs
    for epoch in range(epochs):

        # training
        train_loss = train(model, train_ds)
        train_loss_agg.append(train_loss)

        # testing
        test_metric = test(model, test_ds)
        metrics_agg.append(test_metric)

        # print loss if show flag is set
        if (show):
            print_status(epoch + 1, train_loss, test_metric)

    # return aggregated training and test losses
    return train_loss_agg, metrics_agg


def train(model, ds, training=True):
    """
    Performs training on a full dataset and aggregates the losses

    Args:
        model: the model to train
        ds: the train dataset
        training: flag stating if in training mode

    Returns:
        mean loss
    """

    # loss aggregator
    loss_agg = []

    # for input and target in dataset
    for x, t in ds:

        # calculate loss
        loss = train_step(model, x, t, training)

        # append loss
        loss_agg.append(loss.numpy())

    # calculate mean of losses
    loss = np.mean(loss_agg)

    # reaturn mean loss
    return loss


def train_step(model, x, t, training):
    """
    Peforms a training step by doing a forward and backward pass
    through the network

    Args:
        model: the model to train
        x: input
        t: target output
        training: flag stating if in training mode

    Returns:
        calculated loss
    """

    with GradientTape() as tape:

        # forward step
        prediction = model(x, training)

        # calculate loss
        loss = model.loss(t, prediction)

        # calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)

    # update weights by applying the gradients
    if (training):
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # return calculated loss
    return loss


def test(model, ds, training=False):
    """
    Tests performance on a full dataset and aggregates the metrics

    Args:
        model: the model to train
        ds: the train dataset
        training: flag stating if in training mode

    Returns:
        metrics
    """

    # reset metrics
    for metric in model.compiled_metrics._metrics:
        metric.reset_state()

    # for input and target in dataset
    for x, t in ds:

        # prediction for input
        prediction = model(x, training=training)

        # update states
        for metric in model.compiled_metrics._metrics:
            metric.update_state(t, prediction)

    # compute metrics
    metric_values = { 
        metric.name: metric.result().numpy() for metric in model.compiled_metrics._metrics
    }

    # reaturn mean loss
    return metric_values

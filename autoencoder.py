
from tensorflow.data import Dataset

from preprocessing import prepare_data_autoencoder
from training import train_model
from model.sparse_ae import build_sparse_ae as build_ae
#from model.convolutional_ae import build_convolutional_ae as build_ae
from model.categorical_cf import build_categorical_cf


def autoencoder_classification(data, labels, feature_dim=16):
    """
    Performs autoencoder feature extraction
    and classification

    Args:
        data: splitted data
        labels: class labels
        feature_dim: number of features
    """

    # apply preprocessing
    data, labels = prepare_data_autoencoder(data, labels)

    # extract splitted data
    train_data, test_data, val_data = data
    train_labels, test_labels, val_labels = labels


    # create dataset for autoencoder
    ae_train_ds = Dataset.from_tensor_slices((train_data, train_data))
    ae_test_ds = Dataset.from_tensor_slices((test_data, test_data))

    # shuffle, batch and prefetch
    ae_train_ds = ae_train_ds.shuffle(len(train_data)).batch(32).prefetch(1)
    ae_test_ds = ae_test_ds.shuffle(len(test_data)).batch(32).prefetch(1)

    # build autoencoder model
    autoencoder = build_ae(
        input_dim=train_data.shape[-1],
        latent_dim=feature_dim
    )

    # train autoencoder
    ae_train_loss, ae_metrics = train_model(
        autoencoder, ae_train_ds, ae_test_ds, epochs=200
    )


    # extract encoder from autoencoder
    encoder = autoencoder.layers[0]

    # apply trained autoencoder on datasets
    cf_train_data = encoder.predict(train_data)
    cf_test_data = encoder.predict(test_data)

    # create dataset for classifier
    cf_train_ds = Dataset.from_tensor_slices((cf_train_data, train_labels))
    cf_test_ds = Dataset.from_tensor_slices((cf_test_data, test_labels))

    # shuffle, batch and prefetch
    cf_train_ds = cf_train_ds.shuffle(len(train_data)).batch(32).prefetch(1)
    cf_test_ds = cf_test_ds.shuffle(len(test_data)).batch(32).prefetch(1)

    # build classifier model
    classifier = build_categorical_cf(2)

    # train classifier model
    cf_train_loss, cf_metrics = train_model(
        classifier, cf_train_ds, cf_test_ds, epochs=1000
    )

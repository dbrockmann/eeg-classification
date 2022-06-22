
from tensorflow.data import Dataset

from dataset import load_dataset
from preprocessing import prepare_data_autoencoder, prepare_data_features

from training import train_model

from model.sparse_ae import build_sparse_ae
from model.categorical_cf import build_categorical_cf


# load dataset
X, y = load_dataset('./data/')

# apply preprocessing
data, labels = prepare_data_autoencoder(X, y)
#features_data, features_labels = prepare_data_features(X, y, 16)

# cut last feature
for i, a in enumerate(data):
    data[i] = a[:, :-1]

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
autoencoder = build_sparse_ae(
    input_dim=train_data.shape[-1],
    latent_dim=8
)

# train autoencoder
ae_train_loss, ae_metrics = train_model(
    autoencoder, ae_train_ds, ae_test_ds, epochs=10
)


# extract encoder from autoencoder
encoder = autoencoder.layers[0]

# apply trained autoencoder on datasets
cf_train_data = encoder.predict(train_data)
cf_test_data = encoder.predict(test_data)

# create dataset for autoencoder
cf_train_ds = Dataset.from_tensor_slices((cf_train_data, train_labels))
cf_test_ds = Dataset.from_tensor_slices((cf_test_data, test_labels))

# shuffle, batch and prefetch
cf_train_ds = cf_train_ds.shuffle(len(train_data)).batch(32).prefetch(1)
cf_test_ds = cf_test_ds.shuffle(len(test_data)).batch(32).prefetch(1)

# build classifier model
classifier = build_categorical_cf(2)

# train classifier model
cf_train_loss, cf_metrics = train_model(
    classifier, cf_train_ds, cf_test_ds, epochs=20
)


from tensorflow.data import Dataset

from dataset import load_dataset
from preprocessing import prepare_data
from model import Model_From_Layers
from training import train_from_config

from config.autoencoder import sparse_ae, convolutional_ae
from config.classifier import binary_cf

# load dataset
X, y = load_dataset('./data/')

# apply preprocessing
data, labels = prepare_data(X, y)

# cut last feature
for i, a in enumerate(data):
    data[i] = a[:, :-1]

# extract splitted data
train_data, test_data, val_data = data


# create tensorflow dataset for autoencoder
ae_train_ds = Dataset.from_tensor_slices((train_data, train_data))
ae_test_ds = Dataset.from_tensor_slices((test_data, test_data))

# create autoencoder model
encoder = Model_From_Layers(sparse_ae['encoder'])
decoder = Model_From_Layers(sparse_ae['decoder'])
autoencoder = Model_From_Layers([encoder, decoder])

# train autoencoder model
ae_train_loss, ae_test_loss = train_from_config(
    autoencoder, ae_train_ds, ae_test_ds, sparse_ae['training'], show=True
)


# apply trained autoencoder on datasets
cf_train_data = autoencoder.predict(train_data)
cf_test_data = autoencoder.predict(test_data)

# extract labels
train_labels, test_labels, val_labels = labels

# create tensorflow dataset for classifier
cf_train_ds = Dataset.from_tensor_slices((cf_train_data, train_labels))
cf_test_ds = Dataset.from_tensor_slices((cf_test_data, test_labels))

# create classifier model
classifier = Model_From_Layers(binary_cf['model'])

# train classifier
cf_train_loss, cf_test_loss = train_from_config(
    classifier, cf_train_ds, cf_test_ds, binary_cf['training'], show=True
)

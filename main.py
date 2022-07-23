
from dataset import load_dataset
from preprocessing import split_dataset
from autoencoder import autoencoder_classification
from handcrafted import handcrafted_classification

# load dataset
X, y = load_dataset('./data/')

# split the data in a train, test and validation set in a 70:15:15 ratio
data, labels = split_dataset(X, y)


ae_hist, cf_hist = autoencoder_classification(data, labels, feature_dim=16)

exit()

# number of features to extract
feature_dims = [2, 4, 8, 16, 32]

# perform autoencoder feature extraction
# and classification
for feature_dim in feature_dims:
    ae_hist, cf_hist = autoencoder_classification(data, labels, feature_dim=feature_dim, show=False)
    print('autoencoder', feature_dim, ':', cf_hist[1][-1])

# perform handcrafted feature extraction
# and classification
for feature_dim in feature_dims:
    hist = handcrafted_classification(data, labels, feature_dim=feature_dim, show=False)
    print('handcrafted', feature_dim, ':', hist[1][-1])

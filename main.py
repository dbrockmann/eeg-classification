from dataset import *
from preprocessing import *

X, y = load_dataset('./data/')
data, labels = prepare_data(X, y)

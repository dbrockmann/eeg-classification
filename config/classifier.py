
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense

softmax_cf = {

    'training': {
        'batch_size': 32,
        'loss_function': BinaryCrossentropy(),
        'optimizer': Adam(lr=0.001),
        'epochs': 10
    },

    'model': [
        Dense(
            32, activation='relu'
        ),
        Dense(
            16, activation='relu'
        ),
        Dense(
            1, activation='softmax'
        )
    ]

}


from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D
from tensorflow.keras.regularizers import L1

sparse_ae = {

    'training': {
        'batch_size': 32,
        'loss_function': MeanSquaredError(),
        'optimizer': Adam(lr=0.001),
        'epochs': 20
    },

    'encoder': [
        Dense(
            128, activation='relu'
        ),
        Dense(
            64, activation='relu'
        ),
        Dense(
            8, activation='sigmoid',
            activity_regularizer=L1(0.001)
        )
    ],

    'decoder': [
        Dense(
            64, activation='relu'
        ),
        Dense(
            128, activation='relu'
        ),
        Dense(
            240, activation='sigmoid'
        )
    ]

}

convolutional_ae = {

    'batch_size': 32,
    'loss_function': MeanSquaredError(),
    'optimizer': Adam(lr=0.001),
    'epochs': 20,

    'encoder': [
        Reshape(
            (-1, 1)
        ),
        Conv1D(
            16, kernel_size=3, padding='same', activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Conv1D(
            32, kernel_size=3, padding='same', activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Conv1D(
            64, kernel_size=3, padding='same', activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Flatten(),
        Dense(
            8, activation='relu',
            activity_regularizer=L1(0.001)
        )
    ],

    'decoder': [
        Dense(
            240 // 2**3 * 64
        ),
        Reshape(
            (240 // 2**3, 64)
        ),
        Conv1DTranspose(
            64, kernel_size=3, padding='same', activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1DTranspose(
            32, kernel_size=3, padding='same', activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1DTranspose(
            16, kernel_size=3, padding='same', activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1D(
            1, kernel_size=3, padding='same', activation='sigmoid'
        ),
        Reshape(
            (240,)
        )
    ]

}

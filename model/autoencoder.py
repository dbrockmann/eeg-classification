
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D
from tensorflow.keras.regularizers import L1

sparse_ae = {

    'training': {
        'batch_size': 32,
        'loss_function': MeanSquaredError(),
        'optimizer': Adam(
            learning_rate=0.001
        ),
        'epochs': 20
    },

    'encoder': Sequential([
        Dense(
            units=128, 
            activation='relu'
        ),
        Dense(
            units=64, 
            activation='relu'
        ),
        Dense(
            units=8, 
            activation='sigmoid',
            activity_regularizer=L1(
                l1=0.001
            )
        )
    ]),

    'decoder': Sequential([
        Dense(
            units=64, 
            activation='relu'
        ),
        Dense(
            units=128, 
            activation='relu'
        ),
        Dense(
            units=240, 
            activation='sigmoid'
        )
    ])

}

convolutional_ae = {

    'batch_size': 32,
    'loss_function': MeanSquaredError(),
    'optimizer': Adam(
        learning_rate=0.001
    ),
    'epochs': 20,

    'encoder': Sequential([
        Reshape(
            target_shape=(-1, 1)
        ),
        Conv1D(
            filters=16, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Conv1D(
            filters=32, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Conv1D(
            filters=64, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Flatten(),
        Dense(
            units=8, 
            activation='relu',
            activity_regularizer=L1(
                l1=0.001
            )
        )
    ]),

    'decoder': Sequential([
        Dense(
            units=240 // 2**3 * 64
        ),
        Reshape(
            target_shape=(240 // 2**3, 64)
        ),
        Conv1DTranspose(
            filters=64, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1DTranspose(
            filters=32, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1DTranspose(
            filters=16, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1D(
            filters=1, 
            kernel_size=3, 
            padding='same', 
            activation='sigmoid'
        )
    ])

}

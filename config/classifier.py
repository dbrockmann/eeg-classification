
import tensorflow as tf

softmax_cf = {

    'batch_size': 32,
    'loss_function': tf.keras.losses.BinaryCrossentropy(),
    'optimizer': tf.keras.optimizers.Adam(lr=0.001),
    'epochs': 10,

    'classifier': [
        tf.keras.layers.Dense(
            32, activation='relu'
        ),
        tf.keras.layers.Dense(
            16, activation='relu'
        ),
        tf.keras.layers.Dense(
            1, activation='softmax'
        )
    ]

}

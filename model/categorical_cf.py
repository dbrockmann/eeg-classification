
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseCategoricalCrossentropy as SparseCategoricalCrossentropyMetric


class BalancedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """
    Sparse categorical accuracy that 
    balances the accuracy by weighting with the
    occurence of each label
    """

    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        """
        Initialize metric

        Args:
            name: name of this metric
        """

        super(BalancedSparseCategoricalAccuracy, self).__init__(
            name=name, dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update state with sample weight calculated from 
        the labels

        Args:
            y_true: ground truth label values
            y_pred: predicted probability values
            sample_weight: coefficient for metric
        """

        # calculate weights according to occurence
        if not sample_weight:
            y_true_int = tf.cast(y_true, tf.int32)
            counts = tf.math.bincount(y_true_int)
            weights = tf.math.reciprocal_no_nan(tf.cast(counts, self.dtype))
            sample_weight = tf.gather(weights, y_true_int)
        print(sample_weight)

        # update state with weights
        super(BalancedSparseCategoricalAccuracy, self).update_state(
            y_true, y_pred, sample_weight=sample_weight
        )


def build_categorical_cf(class_dim):
    """
    Build categorical classifier

    Args:
        class_dim: number of classes
    """

    categorical_cf = Sequential([
        Dense(
            units=32, 
            activation='relu'
        ),
        Dense(
            units=16, 
            activation='relu'
        ),
        Dense(
            units=class_dim, 
            activation='softmax'
        )
    ])

    categorical_cf.compile(
        optimizer=Adam(
            learning_rate=0.0001
        ),
        loss=SparseCategoricalCrossentropy(),
        metrics=[
            SparseCategoricalCrossentropyMetric(name='Loss'),
            SparseCategoricalAccuracy(name='Accuracy'),
            BalancedSparseCategoricalAccuracy(name='Balanced Accuracy')
        ]
    )

    return categorical_cf

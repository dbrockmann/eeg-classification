
import numpy as np
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

        # aggregate states
        self.y_true = None
        self.y_pred = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update state

        Args:
            y_true: ground truth label values
            y_pred: predicted probability values
            sample_weight: coefficient for metric
        """

        # concat states
        if self.y_true is not None:
            self.y_true = tf.concat([self.y_true, y_true], 0)
            self.y_pred = tf.concat([self.y_pred, y_pred], 0)

        # initialize with state
        else:
            self.y_true = y_true
            self.y_pred = y_pred

    def reset_state(self):
        """
        Reset state
        """

        self.y_true = None
        self.y_pred = None

        super(BalancedSparseCategoricalAccuracy, self).reset_state()

    def result(self):
        """
        Calculate result with sample weight calculated from 
        the labels

        Returns:
            resulting metric
        """

        # calculate weights according to occurence
        y_true_int = tf.cast(self.y_true, tf.int32)
        counts = tf.math.bincount(y_true_int)
        weights = tf.math.reciprocal_no_nan(tf.cast(counts, self.dtype))
        sample_weight = tf.gather(weights, y_true_int)

        # update state with weights
        super(BalancedSparseCategoricalAccuracy, self).update_state(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        # calculate result
        result = super(BalancedSparseCategoricalAccuracy, self).result()

        return result


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

#!/usr/bin/env python3
"""Basic CNN model."""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class BasicCNNModel:
    """BasicCNNModel."""

    def build_basiccnn_model(self, input_shape=(52, 2)):
        """Builds the CNN model."""

        inputs = layers.Input(shape=input_shape)

        x = layers.Conv1D(32, 3, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv1D(64, 3, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv1D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(2, activation="linear")(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def euclidean_distance_loss(self, y_true, y_pred):
        """Euclidean distance loss."""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
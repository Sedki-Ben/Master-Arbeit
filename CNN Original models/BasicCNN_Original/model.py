#!/usr/bin/env python3
"""Model architecture."""

import tensorflow as tf
from tensorflow.keras import layers, Model

class BasicCNNModel:
    """BasicCNNModel."""

    def build_basic_cnn_model(self, input_shape=(52, 2)):
        """Build BasicCNN Original model architecture"""
        
        
                # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # CNN layers
        x = layers.Conv1D(32, 3, activation="relu")(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation="relu")(x)
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        
        # Dense layers
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(2, activation="linear")(x)
        
        
        model = Model(inputs=inputs, outputs=outputs)
        
        
        
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
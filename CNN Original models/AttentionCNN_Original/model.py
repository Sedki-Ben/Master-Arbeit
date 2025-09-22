#!/usr/bin/env python3
"""Model architecture"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionCNNModel:
    """AttentionCNNModel."""

    def build_attentioncnn_model(self, input_shape=(52, 2)):
        """Build AttentionCNN_Original model architecture"""
        
        
                # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Convolutional layers
        x = layers.Conv1D(64, 3, activation="relu")(inputs)
        x = layers.Conv1D(128, 3, activation="relu")(x)
        
        # Self-attention mechanism
        attention_weights = layers.Dense(x.shape[-1], activation="softmax")(x)
        attended_features = layers.multiply([x, attention_weights])
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(attended_features)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(2, activation="linear")(x)
        
        
        model = Model(inputs=inputs, outputs=outputs, name=f'{model_name}')
        
        
        
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

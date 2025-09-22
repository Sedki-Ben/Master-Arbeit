#!/usr/bin/env python3
"""Model architecture."""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class ResidualCNNModel:
    """ResidualCNNModel."""

    def build_residualcnn_model(self, input_shape=(52, 2)):
        """Build ResidualCNN_model architecture"""
        
        
                # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # First conv block
        x = layers.Conv1D(64, 3, activation="relu", padding='same')(inputs)
        x = layers.BatchNormalization(name="bn_1")(x)
        
        # Residual blocks
        for i in range(3):
            residual = x
            x = layers.Conv1D(64, 3, activation="relu", padding='same', name=f'res_conv_{i}_1')(x)
            x = layers.BatchNormalization(name=f'res_bn_{i}_1')(x)
            x = layers.Conv1D(64, 3, activation="relu", padding='same', name=f'res_conv_{i}_2')(x)
            x = layers.BatchNormalization(name=f'res_bn_{i}_2')(x)
            x = layers.add([x, residual], name=f'residual_add_{i}')
            x = layers.Activation('relu', name=f'residual_relu_{i}')(x)
        
        # Final layers
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
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
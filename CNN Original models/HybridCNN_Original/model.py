#!/usr/bin/env python3
"""Model architecture."""

import tensorflow as tf
from tensorflow.keras import layers, Model

class HybridCNNModel:
    """HybridCNNModel."""

    def build_hybridcnn_model(self, input_shape=(52, 2)):
        """Build HybridCNN_Original model architecture"""
        
        
                # Dual inputs for HybridCNN
        csi_input = layers.Input(shape=input_shape)
        rssi_input = layers.Input(shape=(1,))
        
        # CSI input processing
        csi_branch = layers.Conv1D(32, 3, activation="relu")(csi_input)
        csi_branch = layers.MaxPooling1D(2)(csi_branch)
        csi_branch = layers.Conv1D(64, 3, activation="relu")(csi_branch)
        csi_branch = layers.MaxPooling1D(2)(csi_branch)
        csi_branch = layers.Conv1D(128, 3, activation="relu")(csi_branch)
        csi_branch = layers.GlobalAveragePooling1D(name="csi_global_avg_pool")(csi_branch)
        
        # RSSI input processing
        rssi_branch = layers.Dense(32, activation="relu")(rssi_input)
        rssi_branch = layers.Dense(16, activation="relu")(rssi_branch)
        
        # Combine branches
        combined = layers.concatenate([csi_branch, rssi_branch])
        
        # Final layers
        x = layers.Dense(256, activation="relu")(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(2, activation="linear")(x)
        
         with dual inputs
        model = Model(inputs=[csi_input, rssi_input], outputs=outputs, name=f'{model_name}')
        
        
        
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
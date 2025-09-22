#!/usr/bin/env python3
"""
HybridCNN_Improved - Model Architecture
===========================================

Defines the HybridCNN_Improved model architecture for indoor localization.
HybridCNN with Tom Cruise improvements.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class HybridCNNModel:
    """HybridCNN_Improved model for CSI-based indoor localization"""
    
    def __init__(self):
        print("🧠 HybridCNN_Improved Model initialized")
    
    def build_hybridcnn_model(self, input_shape=(52, 2)):
        """Build HybridCNN_Improved model architecture"""
        
        print(f"🏗️ Building HybridCNN_Improved model")
        print(f"   Input shape: {input_shape}")
        
        # Dual inputs for HybridCNN
        csi_input = layers.Input(shape=input_shape, name='csi_input')
        rssi_input = layers.Input(shape=(1,), name='rssi_input')
        
        # CSI input processing
        csi_branch = layers.Conv1D(32, 3, activation='relu', name='csi_conv1d_1')(csi_input)
        csi_branch = layers.MaxPooling1D(2, name='csi_maxpool1d_1')(csi_branch)
        csi_branch = layers.Conv1D(64, 3, activation='relu', name='csi_conv1d_2')(csi_branch)
        csi_branch = layers.MaxPooling1D(2, name='csi_maxpool1d_2')(csi_branch)
        csi_branch = layers.Conv1D(128, 3, activation='relu', name='csi_conv1d_3')(csi_branch)
        csi_branch = layers.GlobalAveragePooling1D(name='csi_global_avg_pool')(csi_branch)
        
        # RSSI input processing
        rssi_branch = layers.Dense(32, activation='relu', name='rssi_dense_1')(rssi_input)
        rssi_branch = layers.Dense(16, activation='relu', name='rssi_dense_2')(rssi_branch)
        
        # Combine branches
        combined = layers.concatenate([csi_branch, rssi_branch], name='feature_fusion')
        
        # Final processing
        x = layers.Dense(256, activation='relu', name='dense_1')(combined)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='linear', name='coordinates_output')(x)
        
        # Create model with dual inputs
        model = Model(inputs=[csi_input, rssi_input], outputs=outputs, name=f'{model_name}')
        
        print(f"✅ HybridCNN_Improved model created")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

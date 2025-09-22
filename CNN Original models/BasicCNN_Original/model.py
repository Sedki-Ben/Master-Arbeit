#!/usr/bin/env python3
"""
BasicCNN Original - Model Architecture
======================================

Defines the BasicCNN Original model architecture for indoor localization.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class BasicCNNModel:
    """BasicCNN Original model for CSI-based indoor localization"""
    
    def __init__(self):
        print("🧠 BasicCNN Original Model initialized")
    
    def build_basic_cnn_model(self, input_shape=(52, 2)):
        """Build BasicCNN Original model architecture"""
        
        print(f"🏗️ Building BasicCNN Original model")
        print(f"   Input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name='csi_input')
        
        # Original BasicCNN architecture
        x = layers.Conv1D(32, 3, activation='relu', name='conv1d_1')(inputs)
        x = layers.MaxPooling1D(2, name='maxpool1d_1')(x)
        x = layers.Conv1D(64, 3, activation='relu', name='conv1d_2')(x)
        x = layers.MaxPooling1D(2, name='maxpool1d_2')(x)
        x = layers.Conv1D(128, 3, activation='relu', name='conv1d_3')(x)
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='linear', name='coordinates_output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='BasicCNN_Original')
        
        print(f"✅ BasicCNN Original model created")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

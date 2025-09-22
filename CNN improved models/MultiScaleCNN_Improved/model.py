#!/usr/bin/env python3
"""
MultiScaleCNN_Improved - Model Architecture
===============================================

Defines the MultiScaleCNN_Improved model architecture for indoor localization.
MultiScaleCNN with Tom Cruise improvements.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class MultiScaleCNNModel:
    """MultiScaleCNN_Improved model for CSI-based indoor localization"""
    
    def __init__(self):
        print("🧠 MultiScaleCNN_Improved Model initialized")
    
    def build_multiscalecnn_model(self, input_shape=(52, 2)):
        """Build MultiScaleCNN_Improved model architecture"""
        
        print(f"🏗️ Building MultiScaleCNN_Improved model")
        print(f"   Input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name='csi_input')
        
        # Multi-scale convolutional branches
        conv3 = layers.Conv1D(32, 3, activation='relu', padding='same', name='conv1d_3')(inputs)
        conv7 = layers.Conv1D(32, 7, activation='relu', padding='same', name='conv1d_7')(inputs)
        conv15 = layers.Conv1D(32, 15, activation='relu', padding='same', name='conv1d_15')(inputs)
        
        # Concatenate multi-scale features
        multi_scale = layers.concatenate([conv3, conv7, conv15], name='multi_scale_concat')
        
        # Further processing
        x = layers.Conv1D(128, 3, activation='relu', name='conv1d_fusion')(multi_scale)
        x = layers.MaxPooling1D(2, name='maxpool1d_1')(x)
        x = layers.Conv1D(256, 3, activation='relu', name='conv1d_final')(x)
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='linear', name='coordinates_output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=f'{model_name}')
        
        print(f"✅ MultiScaleCNN_Improved model created")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

#!/usr/bin/env python3
"""
BasicCNN_Improved - Model Architecture
==========================================

Defines the BasicCNN_Improved model architecture for indoor localization.
BasicCNN with Tom Cruise improvements.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class BasicCNNModel:
    """BasicCNN_Improved model for CSI-based indoor localization"""
    
    def __init__(self):
        print("🧠 BasicCNN_Improved Model initialized")
    
    def build_basiccnn_model(self, input_shape=(52, 2)):
        """Build BasicCNN_Improved model architecture"""
        
        print(f"🏗️ Building BasicCNN_Improved model")
        print(f"   Input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name='csi_input')
        
        # Improved BasicCNN architecture with regularization
        x = layers.Conv1D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='conv1d_1')(inputs)
        x = layers.MaxPooling1D(2, name='maxpool1d_1')(x)
        x = layers.Dropout(0.3, name='dropout_conv_1')(x)
        
        x = layers.Conv1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='conv1d_2')(x)
        x = layers.MaxPooling1D(2, name='maxpool1d_2')(x)
        x = layers.Dropout(0.3, name='dropout_conv_2')(x)
        
        x = layers.Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='conv1d_3')(x)
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = layers.Dropout(0.4, name='dropout_global')(x)
        
        # Dense layers with regularization
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='dense_1')(x)
        x = layers.Dropout(0.5, name='dropout_dense_1')(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='dense_2')(x)
        x = layers.Dropout(0.4, name='dropout_dense_2')(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='linear', name='coordinates_output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=f'{model_name}')
        
        print(f"✅ BasicCNN_Improved model created")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

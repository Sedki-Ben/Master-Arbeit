#!/usr/bin/env python3
"""
ResidualCNN_Original - Model Architecture
=============================================

Defines the ResidualCNN_Original model architecture for indoor localization.
ResidualCNN with skip connections.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class ResidualCNNModel:
    """ResidualCNN_Original model for CSI-based indoor localization"""
    
    def __init__(self):
        print("🧠 ResidualCNN_Original Model initialized")
    
    def build_residualcnn_model(self, input_shape=(52, 2)):
        """Build ResidualCNN_Original model architecture"""
        
        print(f"🏗️ Building ResidualCNN_Original model")
        print(f"   Input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name='csi_input')
        
        # First conv block
        x = layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        
        # Residual blocks
        for i in range(3):
            residual = x
            x = layers.Conv1D(64, 3, activation='relu', padding='same', name=f'res_conv_{i}_1')(x)
            x = layers.BatchNormalization(name=f'res_bn_{i}_1')(x)
            x = layers.Conv1D(64, 3, activation='relu', padding='same', name=f'res_conv_{i}_2')(x)
            x = layers.BatchNormalization(name=f'res_bn_{i}_2')(x)
            x = layers.add([x, residual], name=f'residual_add_{i}')
            x = layers.Activation('relu', name=f'residual_relu_{i}')(x)
        
        # Final processing
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='linear', name='coordinates_output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=f'{model_name}')
        
        print(f"✅ ResidualCNN_Original model created")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

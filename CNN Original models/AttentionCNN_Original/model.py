#!/usr/bin/env python3
"""
AttentionCNN_Original - Model Architecture
==============================================

Defines the AttentionCNN_Original model architecture for indoor localization.
AttentionCNN with self-attention mechanism.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionCNNModel:
    """AttentionCNN_Original model for CSI-based indoor localization"""
    
    def __init__(self):
        print("🧠 AttentionCNN_Original Model initialized")
    
    def build_attentioncnn_model(self, input_shape=(52, 2)):
        """Build AttentionCNN_Original model architecture"""
        
        print(f"🏗️ Building AttentionCNN_Original model")
        print(f"   Input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name='csi_input')
        
        # Convolutional layers
        x = layers.Conv1D(64, 3, activation='relu', name='conv1d_1')(inputs)
        x = layers.Conv1D(128, 3, activation='relu', name='conv1d_2')(x)
        
        # Self-attention mechanism
        attention_weights = layers.Dense(x.shape[-1], activation='softmax', name='attention_weights')(x)
        attended_features = layers.multiply([x, attention_weights], name='attention_applied')
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(attended_features)
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='linear', name='coordinates_output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=f'{model_name}')
        
        print(f"✅ AttentionCNN_Original model created")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

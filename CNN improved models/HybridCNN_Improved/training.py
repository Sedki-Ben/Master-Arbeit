#!/usr/bin/env python3
"""
HybridCNN_Improved - Training Module
===========================================

Training functionality for HybridCNN_Improved model.
HybridCNN with Tom Cruise improvements.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import time

class HybridCNNTrainer:
    """Model trainer for HybridCNN_Improved"""
    
    def __init__(self, output_dir="hybridcnn_improved_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.history = None
        self.training_time = 0
        
        print(f"🏋️ HybridCNN_Improved ModelTrainer initialized")
        print(f"📁 Model checkpoints will be saved to: {self.output_dir}")
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
    
    def compile_and_train_model(self, model, X_train, y_train, X_val, y_val, dataset_size, model_name="HybridCNN_Improved"):
        """Compile and train model with improved configuration"""
        
        print(f"🚀 Training {model_name} (improved configuration)")
        print(f"📊 Dataset size: {dataset_size}")
        
        # Improved "Tom Cruise" configuration
        learning_rate = 0.0002
        loss_function = 'mse'
        batch_size = 16
        epochs = 150
        patience = 20
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])
        
        print(f"⚙️ Training configuration:")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss function: {loss_function}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max epochs: {epochs}")
        print(f"   Patience: {patience}")
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7, verbose=1),
            ModelCheckpoint(
                filepath=str(self.output_dir / f"{model_name}_{dataset_size}_samples.h5"),
                monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1
            )
        ]
        
        # Train model
        print(f"\n🎯 Starting training...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        self.training_time = training_time
        self.history = history
        
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Training summary
        best_epoch = np.argmin(history.history['val_loss'])
        training_summary = {
            'model_name': model_name,
            'dataset_size': dataset_size,
            'training_time': training_time,
            'total_epochs': len(history.history['loss']),
            'best_epoch': best_epoch + 1,
            'final_train_loss': history.history['loss'][best_epoch],
            'final_val_loss': history.history['val_loss'][best_epoch],
            'checkpoint_path': str(self.output_dir / f"{model_name}_{dataset_size}_samples.h5")
        }
        
        return training_summary

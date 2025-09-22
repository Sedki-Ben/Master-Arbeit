#!/usr/bin/env python3
"""
BasicCNN Original - Training Module
===================================

Training functionality for BasicCNN Original model.
Uses original "last samurai" training configuration.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import time

class BasicCNNTrainer:
    """Model trainer for BasicCNN Original"""
    
    def __init__(self, output_dir="basic_cnn_original_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.history = None
        self.training_time = 0
        
        print(f"🏋️ BasicCNN Original ModelTrainer initialized")
        print(f"📁 Model checkpoints will be saved to: {self.output_dir}")
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom Euclidean distance loss function"""
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
    
    def compile_and_train_model(self, model, X_train, y_train, X_val, y_val, dataset_size, model_name="BasicCNN_Original"):
        """Compile and train model with original configuration"""
        
        print(f"🚀 Training {model_name} (original configuration)")
        print(f"📊 Dataset size: {dataset_size}")
        
        # Original "last samurai" configuration
        learning_rate = 0.001
        loss_function = self.euclidean_distance_loss
        batch_size = 32
        epochs = 100
        patience = 10
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=['mae']
        )
        
        print(f"⚙️ Training configuration:")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss function: Euclidean distance")
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
        print(f"📈 Total epochs: {len(history.history['loss'])}")
        
        # Training summary
        best_epoch = np.argmin(history.history['val_loss'])
        final_train_loss = history.history['loss'][best_epoch]
        final_val_loss = history.history['val_loss'][best_epoch]
        
        training_summary = {
            'model_name': model_name,
            'dataset_size': dataset_size,
            'training_time': training_time,
            'total_epochs': len(history.history['loss']),
            'best_epoch': best_epoch + 1,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'checkpoint_path': str(self.output_dir / f"{model_name}_{dataset_size}_samples.h5")
        }
        
        print(f"\n📊 Training Summary:")
        print(f"   Best epoch: {best_epoch + 1}")
        print(f"   Final train loss: {final_train_loss:.4f}")
        print(f"   Final val loss: {final_val_loss:.4f}")
        
        return training_summary

#!/usr/bin/env python3
"""
MultiScaleCNN_- Preprocessing Module

Data preprocessing functionality for MultiScaleCNN_model.
MultiScaleCNN with 
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

class MultiScaleCNNPreprocessor:
    """MultiScaleCNNPreprocessor."""

    def __init__(self):
        self.scalers = {}
        
    
    def preprocess_data(self, train_data, val_data, test_data):
        """Preprocess data using improved approach"""
        
        print(" Preprocessing data (improved approach)")
        
        # Fit scalers only on training data (improved approach)
        self.scalers['amplitude'] = StandardScaler()
        self.scalers['phase'] = StandardScaler()
        self.scalers['rssi'] = StandardScaler()
        
        self.scalers['amplitude'].fit(train_data['amplitudes'])
        self.scalers['phase'].fit(train_data['phases'])
        self.scalers['rssi'].fit(train_data['rssi'].reshape(-1, 1))
        
        # Transform each set
        processed_train = {
            'amplitudes': self.scalers['amplitude'].transform(train_data['amplitudes']).astype(np.float32),
            'phases': self.scalers['phase'].transform(train_data['phases']).astype(np.float32),
            'rssi': self.scalers['rssi'].transform(train_data['rssi'].reshape(-1, 1)).flatten().astype(np.float32),
            'coordinates': train_data['coordinates'].astype(np.float32)
        }
        
        processed_val = {
            'amplitudes': self.scalers['amplitude'].transform(val_data['amplitudes']).astype(np.float32),
            'phases': self.scalers['phase'].transform(val_data['phases']).astype(np.float32),
            'rssi': self.scalers['rssi'].transform(val_data['rssi'].reshape(-1, 1)).flatten().astype(np.float32),
            'coordinates': val_data['coordinates'].astype(np.float32)
        }
        
        processed_test = {
            'amplitudes': self.scalers['amplitude'].transform(test_data['amplitudes']).astype(np.float32),
            'phases': self.scalers['phase'].transform(test_data['phases']).astype(np.float32),
            'rssi': self.scalers['rssi'].transform(test_data['rssi'].reshape(-1, 1)).flatten().astype(np.float32),
            'coordinates': test_data['coordinates'].astype(np.float32)
        }
        
        print(" Preprocessing complete")
        return processed_train, processed_val, processed_test

    def apply_data_augmentation(self, train_data, augmentation_factor=2):
        """Apply data augmentation (disabled for improved variant)"""
        print("ℹ Data augmentation skipped (improved variant)")
        return train_data
#!/usr/bin/env python3
"""
AttentionCNN_Original- Preprocessing Module

Data preprocessing functionality for AttentionCNN_Original model.
AttentionCNN with self-attention mechanism.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

class AttentionCNNPreprocessor:
    """AttentionCNNPreprocessor."""

    def __init__(self):
        self.scalers = {}
        
    
    def preprocess_data(self, train_data, val_data, test_data):
        """Preprocess data using original approach"""
        
        print(" Preprocessing data (original approach)")
        
        # Combine all data for scaling (original approach)
        all_amps = np.vstack([train_data['amplitudes'], val_data['amplitudes'], test_data['amplitudes']])
        all_phases = np.vstack([train_data['phases'], val_data['phases'], test_data['phases']])
        all_rssi = np.concatenate([train_data['rssi'], val_data['rssi'], test_data['rssi']])
        
        # Initialize and fit scalers on all data
        self.scalers['amplitude'] = StandardScaler()
        self.scalers['phase'] = StandardScaler()
        self.scalers['rssi'] = StandardScaler()
        
        self.scalers['amplitude'].fit(all_amps)
        self.scalers['phase'].fit(all_phases)
        self.scalers['rssi'].fit(all_rssi.reshape(-1, 1))
        
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
        """Apply data augmentation to training data"""
        
        print(f" Applying data augmentation (factor: {augmentation_factor})")
        
        amplitudes = train_data['amplitudes']
        phases = train_data['phases']
        rssi = train_data['rssi']
        coordinates = train_data['coordinates']
        
        augmented_amps = [amplitudes]
        augmented_phases = [phases]
        augmented_rssi = [rssi]
        augmented_coords = [coordinates]
        
        for _ in range(augmentation_factor - 1):
            noise_scale_amp = 0.1 * np.std(amplitudes)
            noise_scale_phase = 0.1 * np.std(phases)
            noise_scale_rssi = 0.1 * np.std(rssi)
            
            aug_amps = amplitudes + np.random.normal(0, noise_scale_amp, amplitudes.shape)
            aug_phases = phases + np.random.normal(0, noise_scale_phase, phases.shape)
            aug_rssi = rssi + np.random.normal(0, noise_scale_rssi, rssi.shape)
            
            augmented_amps.append(aug_amps)
            augmented_phases.append(aug_phases)
            augmented_rssi.append(aug_rssi)
            augmented_coords.append(coordinates)
        
        final_data = {
            'amplitudes': np.vstack(augmented_amps).astype(np.float32),
            'phases': np.vstack(augmented_phases).astype(np.float32),
            'rssi': np.concatenate(augmented_rssi).astype(np.float32),
            'coordinates': np.vstack(augmented_coords).astype(np.float32)
        }
        
        print(f" Data augmentation complete: {len(amplitudes)} -> {len(final_data['amplitudes'])} samples")
        return final_data

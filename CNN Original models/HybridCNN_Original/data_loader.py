#!/usr/bin/env python3
"""
HybridCNN_Original -Data Loader Module

Data loading functionality for HybridCNN_Original model.
HybridCNN combining CSI and RSSI.
"""

import numpy as np
import pandas as pd
import json
import csv
from pathlib import Path
import sys

# Import coordinates configuration
sys.path.append('../../..')
from coordinates_config import get_training_points, get_validation_points, get_testing_points

class HybridCNNDataLoader:
    """HybridCNNDataLoader."""

    def __init__(self):
        self.training_points = get_training_points()
        self.validation_points = get_validation_points()
        self.testing_points = get_testing_points()
        
        
        print(f"   Training points: {len(self.training_points)}")
        print(f"   Validation points: {len(self.validation_points)}")
        print(f"   Testing points: {len(self.testing_points)}")
    
    def load_data_by_coordinates(self, dataset_size, point_type="training"):
        """Load CSI data for specific coordinates and dataset size"""
        
        if point_type == "training":
            points = self.training_points
            folder = f"../../../CSI Dataset {dataset_size} Samples"
        elif point_type == "validation":
            points = self.validation_points  
            folder = f"../../../CSI Dataset {dataset_size} Samples"
        elif point_type == "testing":
            points = self.testing_points
            folder = "../../../Testing Points Dataset 750 Samples"
        else:
            raise ValueError("point_type must be 'training', 'validation', or 'testing'")
        
        print(f" Loading {point_type} data from {len(points)} points...")
        
        amplitudes, phases, rssi_values, coordinates = [], [], [], []
        
        for x, y in points:
            file_path = Path(folder) / f"{x},{y}.csv"
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        try:
                            amps = json.loads(row['amplitude'])
                            phases_data = json.loads(row['phase'])
                            rssi = float(row['rssi'])
                            
                            amplitudes.append(amps)
                            phases.append(phases_data)
                            rssi_values.append(rssi)
                            coordinates.append([x, y])
                            
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            continue
        
        amplitudes = np.array(amplitudes, dtype=np.float32)
        phases = np.array(phases, dtype=np.float32)
        rssi_values = np.array(rssi_values, dtype=np.float32)
        coordinates = np.array(coordinates, dtype=np.float32)
        
        print(f" Loaded {len(amplitudes)} samples")
        
        return amplitudes, phases, rssi_values, coordinates
    
    def load_complete_dataset(self, dataset_sizes=[250, 500, 750]):
        """Load complete dataset for all specified sizes"""
        
        complete_data = {}
        
        for size in dataset_sizes:
            train_amps, train_phases, train_rssi, train_coords = self.load_data_by_coordinates(size, "training")
            val_amps, val_phases, val_rssi, val_coords = self.load_data_by_coordinates(size, "validation")
            test_amps, test_phases, test_rssi, test_coords = self.load_data_by_coordinates(750, "testing")
            
            complete_data[size] = {
                'train': {'amplitudes': train_amps, 'phases': train_phases, 'rssi': train_rssi, 'coordinates': train_coords},
                'validation': {'amplitudes': val_amps, 'phases': val_phases, 'rssi': val_rssi, 'coordinates': val_coords},
                'test': {'amplitudes': test_amps, 'phases': test_phases, 'rssi': test_rssi, 'coordinates': test_coords}
            }
        
        return complete_data

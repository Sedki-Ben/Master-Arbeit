#!/usr/bin/env python3
"""
Classical Models - Shared Data Loader
=====================================

Shared data loading utilities for classical localization models.
Handles loading and preprocessing of CSI amplitude, phase, and RSSI data.
"""

import numpy as np
import pandas as pd
import json
import csv
from pathlib import Path
import sys
import glob
import os

# Adjust path for coordinates_config
sys.path.append(str(Path(__file__).resolve().parents[3]))
from coordinates_config import get_training_points, get_validation_points, get_testing_points

def load_amplitude_phase_data(data_source="amplitude_phase_single"):
    """
    Load amplitude and phase data from single CSV files
    
    Args:
        data_source (str): Data source type ("amplitude_phase_single" or "csi_dataset")
        
    Returns:
        tuple: (X, y, coordinates) where X is features, y is coordinates, coordinates is unique points
    """
    
    print("📂 Loading Amplitude and Phase Data...")
    
    if data_source == "amplitude_phase_single":
        # Load from Amplitude Phase Data Single folder (legacy format)
        return _load_from_amplitude_phase_single()
    elif data_source == "csi_dataset":
        # Load from CSI Dataset folders (current format)
        return _load_from_csi_dataset()
    else:
        raise ValueError("data_source must be 'amplitude_phase_single' or 'csi_dataset'")

def _load_from_amplitude_phase_single():
    """Load data from Amplitude Phase Data Single folder"""
    
    # Get all CSV files from Amplitude Phase Data Single folder
    base_path = Path(__file__).resolve().parents[3]
    data_files = glob.glob(str(base_path / "Amplitude Phase Data Single" / "*.csv"))
    
    all_data = []
    coordinates = []
    
    for file_path in data_files:
        # Extract coordinates from filename (e.g., "0,0.csv" -> (0, 0))
        filename = os.path.basename(file_path)
        coord_str = filename.replace('.csv', '')
        try:
            x, y = map(int, coord_str.split(','))
            coordinates.append((x, y))
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Extract features (amplitude + RSSI)
            features = []
            for _, row in df.iterrows():
                # Parse amplitude array - it's stored as a string representation of a list
                amp_str = row['amplitude'].strip('[]"')
                amplitudes = [float(x.strip()) for x in amp_str.split(',')]
                
                # Add RSSI
                rssi = row['rssi']
                
                # Combine features (52 amplitudes + 1 RSSI = 53 features)
                feature_vector = amplitudes + [rssi]
                features.append(feature_vector)
            
            # Add to dataset
            for feature_vector in features:
                all_data.append({
                    'features': feature_vector,
                    'x': x,
                    'y': y
                })
                
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_data)} samples from {len(coordinates)} reference points")
    
    # Convert to arrays
    X = np.array([item['features'] for item in all_data])
    y = np.array([[item['x'], item['y']] for item in all_data])
    
    return X, y, coordinates

def _load_from_csi_dataset():
    """Load data from CSI Dataset folders (current format)"""
    
    base_path = Path(__file__).resolve().parents[3]
    
    # Use training points for classical models
    training_points = get_training_points()
    
    all_data = []
    coordinates = []
    
    # Load from CSI Dataset 750 Samples folder
    data_folder = base_path / "CSI Dataset 750 Samples"
    
    for x, y in training_points:
        file_path = data_folder / f"{x},{y}.csv"
        
        if file_path.exists():
            coordinates.append((x, y))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Extract amplitude and RSSI
                        amplitudes = json.loads(row['amplitude'])
                        rssi = float(row['rssi'])
                        
                        if len(amplitudes) != 52:
                            continue
                        
                        # Combine features (52 amplitudes + 1 RSSI = 53 features)
                        feature_vector = amplitudes + [rssi]
                        
                        all_data.append({
                            'features': feature_vector,
                            'x': x,
                            'y': y
                        })
                        
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        continue
        else:
            print(f"⚠️ Missing file: {file_path}")
    
    print(f"✅ Loaded {len(all_data)} samples from {len(coordinates)} reference points")
    
    # Convert to arrays
    X = np.array([item['features'] for item in all_data])
    y = np.array([[item['x'], item['y']] for item in all_data])
    
    return X, y, coordinates

def load_test_data():
    """
    Load test data for evaluation
    
    Returns:
        tuple: (X_test, y_test) arrays
    """
    
    print("📂 Loading Test Data...")
    
    base_path = Path(__file__).resolve().parents[3]
    testing_points = get_testing_points()
    
    all_test_data = []
    
    # Load from Testing Points Dataset 750 Samples folder
    test_folder = base_path / "Testing Points Dataset 750 Samples"
    
    for x, y in testing_points:
        file_path = test_folder / f"{x},{y}.csv"
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Extract amplitude and RSSI
                        amplitudes = json.loads(row['amplitude'])
                        rssi = float(row['rssi'])
                        
                        if len(amplitudes) != 52:
                            continue
                        
                        # Combine features (52 amplitudes + 1 RSSI = 53 features)
                        feature_vector = amplitudes + [rssi]
                        
                        all_test_data.append({
                            'features': feature_vector,
                            'x': x,
                            'y': y
                        })
                        
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        continue
        else:
            print(f"⚠️ Missing test file: {file_path}")
    
    print(f"✅ Loaded {len(all_test_data)} test samples")
    
    # Convert to arrays
    X_test = np.array([item['features'] for item in all_test_data])
    y_test = np.array([[item['x'], item['y']] for item in all_test_data])
    
    return X_test, y_test

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train/test split ensuring all reference points are represented
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Coordinate matrix
        test_size (float): Fraction of reference points for testing
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    print(f"📊 Creating Train/Test Split (test_size={test_size})...")
    
    # Get unique coordinates
    unique_coords = np.unique(y, axis=0)
    n_train_points = int((1 - test_size) * len(unique_coords))
    
    np.random.seed(random_state)
    train_coords = unique_coords[np.random.choice(len(unique_coords), n_train_points, replace=False)]
    
    # Create masks for train/test split
    train_mask = np.array([tuple(coord) in [tuple(tc) for tc in train_coords] for coord in y])
    test_mask = ~train_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"   Training: {len(X_train)} samples from {n_train_points} reference points")
    print(f"   Testing: {len(X_test)} samples from {len(unique_coords) - n_train_points} reference points")
    
    return X_train, X_test, y_train, y_test

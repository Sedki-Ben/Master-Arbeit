#!/usr/bin/env python3
"""
Classical Models - Shared Preprocessing

Shared preprocessing utilities for classical localization models.
Handles feature scaling, normalization, and data preparation.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def preprocess_features(X_train, X_test=None, scaler_type='standard', fit_scaler=True, scaler=None):
    """
    Preprocess features using specified scaling method
    
    Args:
        X_train (np.array): Training features
        X_test (np.array, optional): Test features
        scaler_type (str): Type of scaler ('standard', 'minmax', 'robust')
        fit_scaler (bool): Whether to fit a new scaler
        scaler (object, optional): Pre-fitted scaler to use
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler) or (X_train_scaled, scaler) if X_test is None
    """
    
    print(f" Preprocessing features with {scaler_type} scaling...")
    
    if fit_scaler:
        # Create new scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'robust'")
        
        # Fit and transform training data
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        # Use provided scaler
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        X_train_scaled = scaler.transform(X_train)
    
    # Transform test data if provided
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        print(f" Preprocessed {len(X_train_scaled)} training and {len(X_test_scaled)} test samples")
        return X_train_scaled, X_test_scaled, scaler
    else:
        print(f" Preprocessed {len(X_train_scaled)} training samples")
        return X_train_scaled, scaler

def extract_amplitude_features(X):
    """
    Extract amplitude-only features (first 52 features)
    
    Args:
        X (np.array): Full feature matrix (amplitude + RSSI)
        
    Returns:
        np.array: Amplitude features only
    """
    return X[:, :52]

def extract_rssi_features(X):
    """
    Extract RSSI-only features (last feature)
    
    Args:
        X (np.array): Full feature matrix (amplitude + RSSI)
        
    Returns:
        np.array: RSSI features only
    """
    return X[:, -1].reshape(-1, 1)

def combine_features(amplitude_features, rssi_features):
    """
    Combine amplitude and RSSI features
    
    Args:
        amplitude_features (np.array): Amplitude features (N x 52)
        rssi_features (np.array): RSSI features (N x 1)
        
    Returns:
        np.array: Combined features (N x 53)
    """
    return np.hstack([amplitude_features, rssi_features])

def add_statistical_features(X):
    """
    Add statistical features derived from amplitude values
    
    Args:
        X (np.array): Feature matrix with amplitude features
        
    Returns:
        np.array: Extended feature matrix with statistical features
    """
    
    print(" Adding statistical features...")
    
    # Extract amplitude features (first 52 columns)
    amplitudes = X[:, :52]
    
    # Calculate statistical features
    amp_mean = np.mean(amplitudes, axis=1, keepdims=True)
    amp_std = np.std(amplitudes, axis=1, keepdims=True)
    amp_max = np.max(amplitudes, axis=1, keepdims=True)
    amp_min = np.min(amplitudes, axis=1, keepdims=True)
    amp_range = amp_max - amp_min
    amp_median = np.median(amplitudes, axis=1, keepdims=True)
    
    # Spectral features
    amp_q25 = np.percentile(amplitudes, 25, axis=1, keepdims=True)
    amp_q75 = np.percentile(amplitudes, 75, axis=1, keepdims=True)
    amp_iqr = amp_q75 - amp_q25
    
    # Energy-based features
    amp_energy = np.sum(amplitudes**2, axis=1, keepdims=True)
    amp_rms = np.sqrt(amp_energy / amplitudes.shape[1])
    
    # Combine all statistical features
    statistical_features = np.hstack([
        amp_mean, amp_std, amp_max, amp_min, amp_range, amp_median,
        amp_q25, amp_q75, amp_iqr, amp_energy, amp_rms
    ])
    
    # Combine with original features
    extended_features = np.hstack([X, statistical_features])
    
    print(f" Added {statistical_features.shape[1]} statistical features")
    print(f"   Original features: {X.shape[1]}, Extended features: {extended_features.shape[1]}")
    
    return extended_features

def remove_outliers(X, y, method='iqr', threshold=1.5):
    """
    Remove outliers from the dataset
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target coordinates
        method (str): Outlier detection method ('iqr', 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        tuple: (X_clean, y_clean) without outliers
    """
    
    print(f" Removing outliers using {method} method (threshold={threshold})...")
    
    if method == 'iqr':
        # Use IQR method for each feature
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Find samples within bounds for all features
        mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
        
    elif method == 'zscore':
        # Use Z-score method
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        mask = np.all(z_scores < threshold, axis=1)
        
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    outliers_removed = len(X) - len(X_clean)
    print(f" Removed {outliers_removed} outliers ({outliers_removed/len(X)*100:.1f}%)")
    print(f"   Clean dataset: {len(X_clean)} samples")
    
    return X_clean, y_clean

def prepare_classical_features(X_train, X_test=None, include_statistical=False, 
                             scaler_type='standard', remove_outliers_flag=False):
    """
    feature preparation pipeline for classical models
    
    Args:
        X_train (np.array): Training features
        X_test (np.array, optional): Test features
        include_statistical (bool): Whether to add statistical features
        scaler_type (str): Type of scaling to apply
        remove_outliers_flag (bool): Whether to remove outliers
        
    Returns:
        dict: Dictionary containing processed features and metadata
    """
    
    print(" Preparing features for classical models...")
    
    # Store original shapes
    original_train_shape = X_train.shape
    original_test_shape = X_test.shape if X_test is not None else None
    
    # Add statistical features if requested
    if include_statistical:
        X_train = add_statistical_features(X_train)
        if X_test is not None:
            X_test = add_statistical_features(X_test)
    
    # Remove outliers if requested
    if remove_outliers_flag and X_test is None:
        X_train, _ = remove_outliers(X_train, X_train)  # Note: y not used for outlier detection
    
    # Scale features
    if X_test is not None:
        X_train_scaled, X_test_scaled, scaler = preprocess_features(
            X_train, X_test, scaler_type=scaler_type
        )
    else:
        X_train_scaled, scaler = preprocess_features(
            X_train, scaler_type=scaler_type
        )
        X_test_scaled = None
    
    # Prepare result dictionary
    result = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'scaler': scaler,
        'feature_info': {
            'original_train_shape': original_train_shape,
            'original_test_shape': original_test_shape,
            'final_train_shape': X_train_scaled.shape,
            'final_test_shape': X_test_scaled.shape if X_test_scaled is not None else None,
            'statistical_features_added': include_statistical,
            'scaler_type': scaler_type,
            'outliers_removed': remove_outliers_flag
        }
    }
    
    print(f" Feature preparation complete!")
    print(f"   Final training shape: {X_train_scaled.shape}")
    if X_test_scaled is not None:
        print(f"   Final test shape: {X_test_scaled.shape}")
    
    return result
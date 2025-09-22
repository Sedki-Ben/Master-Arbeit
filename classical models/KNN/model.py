#!/usr/bin/env python3
"""
k-Nearest Neighbors (k-NN) Model
================================

k-NN implementation for indoor localization.
Uses Euclidean distance in feature space to find nearest neighbors
and averages their coordinates for position estimation.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNLocalizer:
    """k-Nearest Neighbors localization regressor"""
    
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        """
        Initialize k-NN localizer
        
        Args:
            k (int): Number of neighbors to consider
            distance_metric (str): Distance metric ('euclidean', 'manhattan', 'minkowski')
            weights (str): Weight function ('uniform', 'distance')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.nn_model = None
        
        print(f"🎯 Initializing k-NN Localizer (k={k}, metric={distance_metric}, weights={weights})")
        
    def fit(self, X, y):
        """
        Store training data and fit nearest neighbors model
        
        Args:
            X (np.array): Training features (N x D)
            y (np.array): Training coordinates (N x 2)
        """
        print(f"🔧 Training k-NN model with {len(X)} samples...")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Create and fit nearest neighbors model for efficient search
        self.nn_model = NearestNeighbors(
            n_neighbors=self.k, 
            metric=self.distance_metric
        )
        self.nn_model.fit(X)
        
        print(f"✅ k-NN model trained successfully")
        
    def predict(self, X):
        """
        Predict locations using k-NN regression
        
        Args:
            X (np.array): Test features (M x D)
            
        Returns:
            np.array: Predicted coordinates (M x 2)
        """
        if self.nn_model is None:
            raise ValueError("Model must be fitted before prediction")
            
        print(f"🔮 Predicting locations for {len(X)} test samples...")
        
        predictions = []
        
        # Find k nearest neighbors for each test sample
        distances, indices = self.nn_model.kneighbors(X)
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if self.weights == 'uniform':
                # Simple average of k nearest neighbors
                pred_coord = np.mean(self.y_train[idx], axis=0)
                
            elif self.weights == 'distance':
                # Weighted average based on inverse distance
                # Add small epsilon to avoid division by zero
                weights = 1 / (dist + 1e-8)
                weights = weights / np.sum(weights)  # Normalize weights
                
                pred_coord = np.sum(weights.reshape(-1, 1) * self.y_train[idx], axis=0)
                
            else:
                raise ValueError("weights must be 'uniform' or 'distance'")
                
            predictions.append(pred_coord)
            
        predictions = np.array(predictions)
        print(f"✅ Prediction complete")
        
        return predictions
    
    def get_neighbors_info(self, X, return_distances=True):
        """
        Get detailed information about nearest neighbors
        
        Args:
            X (np.array): Test features
            return_distances (bool): Whether to return distances
            
        Returns:
            dict: Information about nearest neighbors
        """
        if self.nn_model is None:
            raise ValueError("Model must be fitted before neighbor query")
            
        distances, indices = self.nn_model.kneighbors(X)
        
        neighbors_info = {
            'indices': indices,
            'coordinates': self.y_train[indices],
            'features': self.X_train[indices]
        }
        
        if return_distances:
            neighbors_info['distances'] = distances
            
        return neighbors_info
    
    def analyze_neighborhood(self, X_test, sample_idx=0):
        """
        Analyze the neighborhood for a specific test sample
        
        Args:
            X_test (np.array): Test features
            sample_idx (int): Index of sample to analyze
            
        Returns:
            dict: Detailed neighborhood analysis
        """
        if sample_idx >= len(X_test):
            raise ValueError("sample_idx out of bounds")
            
        # Get neighbors for this sample
        test_sample = X_test[sample_idx:sample_idx+1]
        neighbors_info = self.get_neighbors_info(test_sample)
        
        distances = neighbors_info['distances'][0]
        coordinates = neighbors_info['coordinates'][0]
        
        # Calculate statistics
        analysis = {
            'test_sample_idx': sample_idx,
            'k_neighbors': self.k,
            'neighbor_distances': distances,
            'neighbor_coordinates': coordinates,
            'distance_stats': {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances)
            },
            'coordinate_stats': {
                'mean_coord': np.mean(coordinates, axis=0),
                'std_coord': np.std(coordinates, axis=0),
                'coord_range': np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
            }
        }
        
        return analysis

class MultiKNNEnsemble:
    """Ensemble of k-NN models with different k values"""
    
    def __init__(self, k_values=[1, 3, 5, 7, 9], distance_metric='euclidean', ensemble_method='average'):
        """
        Initialize ensemble of k-NN models
        
        Args:
            k_values (list): List of k values to use
            distance_metric (str): Distance metric for all models
            ensemble_method (str): How to combine predictions ('average', 'weighted_average', 'median')
        """
        self.k_values = k_values
        self.distance_metric = distance_metric
        self.ensemble_method = ensemble_method
        self.models = {}
        
        # Create individual k-NN models
        for k in k_values:
            self.models[k] = KNNLocalizer(k=k, distance_metric=distance_metric)
            
        print(f"🎯 Initialized k-NN Ensemble with k values: {k_values}")
        
    def fit(self, X, y):
        """Fit all models in the ensemble"""
        print(f"🔧 Training k-NN Ensemble...")
        
        for k, model in self.models.items():
            print(f"   Training k-NN with k={k}...")
            model.fit(X, y)
            
        print(f"✅ k-NN Ensemble training complete")
        
    def predict(self, X):
        """
        Predict using ensemble of k-NN models
        
        Args:
            X (np.array): Test features
            
        Returns:
            np.array: Ensemble predictions
        """
        print(f"🔮 Ensemble prediction using {len(self.models)} models...")
        
        # Get predictions from all models
        all_predictions = []
        for k, model in self.models.items():
            pred = model.predict(X)
            all_predictions.append(pred)
            
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples, 2)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            ensemble_pred = np.mean(all_predictions, axis=0)
            
        elif self.ensemble_method == 'weighted_average':
            # Weight by inverse of k (favor smaller k values)
            weights = np.array([1.0 / k for k in self.k_values])
            weights = weights / np.sum(weights)  # Normalize
            
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
            
        elif self.ensemble_method == 'median':
            ensemble_pred = np.median(all_predictions, axis=0)
            
        else:
            raise ValueError("ensemble_method must be 'average', 'weighted_average', or 'median'")
            
        print(f"✅ Ensemble prediction complete using {self.ensemble_method}")
        
        return ensemble_pred
    
    def get_individual_predictions(self, X):
        """
        Get predictions from each individual model
        
        Args:
            X (np.array): Test features
            
        Returns:
            dict: Predictions from each k value
        """
        individual_predictions = {}
        
        for k, model in self.models.items():
            pred = model.predict(X)
            individual_predictions[k] = pred
            
        return individual_predictions

#!/usr/bin/env python3
"""
Inverse Distance Weighting (IDW) Model.

IDW implementation for indoor localization.
Uses inverse distance weighting to interpolate coordinates
based on distance in feature space.
"""

import numpy as np
from scipy.spatial.distance import cdist

class IDWLocalizer:
    """IDWLocalizer."""

    def __init__(self, power=2, epsilon=1e-6, distance_metric='euclidean', max_neighbors=None):
        """
        Initialize IDW localizer
        
        Args:
            power (float): Power parameter for inverse distance weighting
            epsilon (float): Small value to avoid division by zero
            distance_metric (str): Distance metric ('euclidean', 'manhattan', 'minkowski')
            max_neighbors (int, optional): Maximum number of neighbors to consider
        """
        self.power = power
        self.epsilon = epsilon
        self.distance_metric = distance_metric
        self.max_neighbors = max_neighbors
        self.X_train = None
        self.y_train = None
        
        print(f" Initializing IDW Localizer (power={power}, metric={distance_metric}, max_neighbors={max_neighbors})")
        
    def fit(self, X, y):
        """
        Store training data
        
        Args:
            X (np.array): Training features (N x D)
            y (np.array): Training coordinates (N x 2)
        """
        print(f" Training IDW model with {len(X)} samples...")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        print(f" IDW model trained successfully")
        
    def predict(self, X):
        """
        Predict locations using IDW interpolation
        
        Args:
            X (np.array): Test features (M x D)
            
        Returns:
            np.array: Predicted coordinates (M x 2)
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        print(f" Predicting locations for {len(X)} test samples...")
        
        predictions = []
        
        for x_test in X:
            # Calculate distances to all training points
            if self.distance_metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            elif self.distance_metric == 'manhattan':
                distances = np.sum(np.abs(self.X_train - x_test), axis=1)
            elif self.distance_metric == 'minkowski':
                # Using p=3 for Minkowski distance
                distances = np.sum(np.abs(self.X_train - x_test)**3, axis=1)**(1/3)
            else:
                raise ValueError("Unsupported distance metric")
            
            # Add small epsilon to avoid division by zero
            distances = distances + self.epsilon
            
            # Limit to max_neighbors if specified
            if self.max_neighbors is not None and len(distances) > self.max_neighbors:
                # Find the k closest neighbors
                nearest_indices = np.argsort(distances)[:self.max_neighbors]
                distances = distances[nearest_indices]
                coords = self.y_train[nearest_indices]
            else:
                coords = self.y_train
            
            # Calculate inverse distance weights
            weights = 1 / (distances ** self.power)
            
            # Handle potential infinity values (exact matches)
            inf_mask = np.isinf(weights)
            if np.any(inf_mask):
                # If we have exact matches, use only those
                exact_coords = coords[inf_mask]
                pred_coord = np.mean(exact_coords, axis=0)
            else:
                # Weighted average of coordinates
                weighted_coords = np.sum(weights.reshape(-1, 1) * coords, axis=0)
                total_weight = np.sum(weights)
                pred_coord = weighted_coords / total_weight
            
            predictions.append(pred_coord)
            
        predictions = np.array(predictions)
        print(f" Prediction complete")
        
        return predictions
    
    def get_weights_info(self, X_test, sample_idx=0):
        """
        Get detailed weighting information for a specific test sample
        
        Args:
            X_test (np.array): Test features
            sample_idx (int): Index of sample to analyze
            
        Returns:
            dict: Detailed weighting analysis
        """
        if sample_idx >= len(X_test):
            raise ValueError("sample_idx out of bounds")
            
        x_test = X_test[sample_idx]
        
        # Calculate distances
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x_test), axis=1)
        else:
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
        
        distances = distances + self.epsilon
        
        # Calculate weights
        weights = 1 / (distances ** self.power)
        weights_normalized = weights / np.sum(weights)
        
        # Sort by weight (highest first)
        sorted_indices = np.argsort(weights)[::-1]
        
        analysis = {
            'test_sample_idx': sample_idx,
            'power': self.power,
            'n_training_points': len(self.X_train),
            'distances': distances,
            'weights': weights,
            'weights_normalized': weights_normalized,
            'sorted_indices': sorted_indices,
            'top_contributors': {
                'indices': sorted_indices[:10],  # Top 10 contributors
                'weights': weights_normalized[sorted_indices[:10]],
                'distances': distances[sorted_indices[:10]],
                'coordinates': self.y_train[sorted_indices[:10]]
            },
            'weight_stats': {
                'max_weight': np.max(weights_normalized),
                'min_weight': np.min(weights_normalized),
                'weight_concentration': np.sum(weights_normalized[:10])  # Weight in top 10
            }
        }
        
        return analysis

class AdaptiveIDWLocalizer:
    """AdaptiveIDWLocalizer."""

    def __init__(self, base_power=2, epsilon=1e-6, adaptation_method='density'):
        """
        Initialize Adaptive IDW localizer
        
        Args:
            base_power (float): Base power parameter
            epsilon (float): Small value to avoid division by zero
            adaptation_method (str): How to adapt power ('density', 'distance', 'variance')
        """
        self.base_power = base_power
        self.epsilon = epsilon
        self.adaptation_method = adaptation_method
        self.X_train = None
        self.y_train = None
        self.local_densities = None
        
        print(f" Initializing Adaptive IDW (base_power={base_power}, adaptation={adaptation_method})")
        
    def fit(self, X, y):
        """
        Store training data and compute local adaptations
        
        Args:
            X (np.array): Training features
            y (np.array): Training coordinates
        """
        print(f" Training Adaptive IDW model with {len(X)} samples...")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Compute local properties for adaptation
        if self.adaptation_method == 'density':
            self._compute_local_densities()
        elif self.adaptation_method == 'variance':
            self._compute_local_variances()
            
        print(f" Adaptive IDW model trained successfully")
        
    def _compute_local_densities(self, k=5):
        """Compute local density for each training point"""
        
        self.local_densities = np.zeros(len(self.X_train))
        
        for i, x in enumerate(self.X_train):
            # Find k nearest neighbors
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_nearest_dist = np.sort(distances)[1:k+1]  # Exclude self (distance 0)
            
            # Density is inverse of average distance to k neighbors
            avg_dist = np.mean(k_nearest_dist)
            self.local_densities[i] = 1 / (avg_dist + self.epsilon)
    
    def _compute_local_variances(self, k=5):
        """Compute local coordinate variance for each training point"""
        
        self.local_variances = np.zeros(len(self.X_train))
        
        for i, x in enumerate(self.X_train):
            # Find k nearest neighbors
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self
            
            # Variance of coordinates among neighbors
            neighbor_coords = self.y_train[k_nearest_indices]
            coord_variance = np.var(neighbor_coords, axis=0)
            self.local_variances[i] = np.mean(coord_variance)
    
    def _adapt_power(self, x_test):
        """
        Adapt power parameter based on local properties
        
        Args:
            x_test (np.array): Test sample
            
        Returns:
            float: Adapted power parameter
        """
        if self.adaptation_method == 'density':
            # Find nearest training points and their densities
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            nearest_idx = np.argmin(distances)
            local_density = self.local_densities[nearest_idx]
            
            # Higher density -> higher power (more local weighting)
            density_factor = local_density / np.mean(self.local_densities)
            adapted_power = self.base_power * (0.5 + 0.5 * density_factor)
            
        elif self.adaptation_method == 'variance':
            # Higher variance -> lower power (more global weighting)
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            nearest_idx = np.argmin(distances)
            local_variance = self.local_variances[nearest_idx]
            
            variance_factor = local_variance / np.mean(self.local_variances)
            adapted_power = self.base_power * (1.5 - 0.5 * variance_factor)
            
        elif self.adaptation_method == 'distance':
            # Adapt based on distance to nearest neighbor
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            min_distance = np.min(distances)
            avg_distance = np.mean(distances)
            
            # Closer to training data -> higher power
            distance_factor = avg_distance / (min_distance + self.epsilon)
            adapted_power = self.base_power * (0.5 + 0.5 * distance_factor)
            
        else:
            adapted_power = self.base_power
            
        # Clamp power to reasonable range
        adapted_power = np.clip(adapted_power, 0.5, 5.0)
        
        return adapted_power
    
    def predict(self, X):
        """
        Predict locations using adaptive IDW
        
        Args:
            X (np.array): Test features
            
        Returns:
            np.array: Predicted coordinates
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        print(f" Predicting with Adaptive IDW for {len(X)} test samples...")
        
        predictions = []
        
        for x_test in X:
            # Adapt power parameter for this test sample
            adapted_power = self._adapt_power(x_test)
            
            # Calculate distances
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            distances = distances + self.epsilon
            
            # Calculate weights with adapted power
            weights = 1 / (distances ** adapted_power)
            
            # Weighted average
            weighted_coords = np.sum(weights.reshape(-1, 1) * self.y_train, axis=0)
            total_weight = np.sum(weights)
            pred_coord = weighted_coords / total_weight
            
            predictions.append(pred_coord)
            
        predictions = np.array(predictions)
        print(f" Adaptive IDW prediction complete")
        
        return predictions

class MultiPowerIDWEnsemble:
    """MultiPowerIDWEnsemble."""

    def __init__(self, power_values=[0.5, 1.0, 1.5, 2.0, 3.0], ensemble_method='average'):
        """
        Initialize ensemble of IDW models
        
        Args:
            power_values (list): List of power values to use
            ensemble_method (str): How to combine predictions ('average', 'weighted_average', 'median')
        """
        self.power_values = power_values
        self.ensemble_method = ensemble_method
        self.models = {}
        
        # Create individual IDW models
        for power in power_values:
            self.models[power] = IDWLocalizer(power=power)
            
        print(f" Initialized IDW Ensemble with power values: {power_values}")
        
    def fit(self, X, y):
        """Fit all models in the ensemble"""
        print(f" Training IDW Ensemble...")
        
        for power, model in self.models.items():
            print(f"   Training IDW with power={power}...")
            model.fit(X, y)
            
        print(f" IDW Ensemble training complete")
        
    def predict(self, X):
        """
        Predict using ensemble of IDW models
        
        Args:
            X (np.array): Test features
            
        Returns:
            np.array: Ensemble predictions
        """
        print(f" Ensemble prediction using {len(self.models)} models...")
        
        # Get predictions from all models
        all_predictions = []
        for power, model in self.models.items():
            pred = model.predict(X)
            all_predictions.append(pred)
            
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples, 2)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            ensemble_pred = np.mean(all_predictions, axis=0)
            
        elif self.ensemble_method == 'weighted_average':
            # Weight by inverse of power (favor higher powers for local accuracy)
            weights = np.array([1.0 / (p + 0.1) for p in self.power_values])
            weights = weights / np.sum(weights)  # Normalize
            
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
            
        elif self.ensemble_method == 'median':
            ensemble_pred = np.median(all_predictions, axis=0)
            
        else:
            raise ValueError("ensemble_method must be 'average', 'weighted_average', or 'median'")
            
        print(f" Ensemble prediction complete using {self.ensemble_method}")
        
        return ensemble_pred

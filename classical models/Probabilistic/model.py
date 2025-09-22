#!/usr/bin/env python3
"""
Probabilistic Fingerprinting Model

Probabilistic localization using Gaussian distributions.
Models CSI signatures at each reference point with multivariate Gaussians
and uses maximum likelihood estimation for localization.
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance
from sklearn.mixture import GaussianMixture

class ProbabilisticLocalizer:
    """ProbabilisticLocalizer."""

    def __init__(self, smoothing=1e-6, covariance_type='empirical', regularization=None):
        """
        Initialize Probabilistic localizer
        
        Args:
            smoothing (float): Smoothing parameter for numerical stability
            covariance_type (str): Type of covariance estimation ('empirical', 'ledoit_wolf', 'oas', 'shrunk')
            regularization (float, optional): Additional regularization for covariance
        """
        self.smoothing = smoothing
        self.covariance_type = covariance_type
        self.regularization = regularization
        self.reference_points = {}
        
        print(f" Initializing Probabilistic Localizer (covariance={covariance_type}, smoothing={smoothing})")
        
    def _estimate_covariance(self, samples):
        """
        Estimate covariance matrix using specified method
        
        Args:
            samples (np.array): Sample data (N x D)
            
        Returns:
            np.array: Covariance matrix (D x D)
        """
        if len(samples) == 1:
            # Single sample - use identity matrix scaled by smoothing
            return self.smoothing * np.eye(samples.shape[1])
        
        if self.covariance_type == 'empirical':
            cov = np.cov(samples, rowvar=False)
            
        elif self.covariance_type == 'ledoit_wolf':
            estimator = LedoitWolf()
            estimator.fit(samples)
            cov = estimator.covariance_
            
        elif self.covariance_type == 'oas':
            estimator = OAS()
            estimator.fit(samples)
            cov = estimator.covariance_
            
        elif self.covariance_type == 'shrunk':
            estimator = ShrunkCovariance()
            estimator.fit(samples)
            cov = estimator.covariance_
            
        else:
            raise ValueError("Unsupported covariance_type")
        
        # Add smoothing to diagonal for numerical stability
        cov += self.smoothing * np.eye(cov.shape[0])
        
        # Additional regularization if specified
        if self.regularization is not None:
            cov += self.regularization * np.eye(cov.shape[0])
        
        return cov
        
    def fit(self, X, y):
        """
        Learn Gaussian distributions for each reference point
        
        Args:
            X (np.array): Training features (N x D)
            y (np.array): Training coordinates (N x 2)
        """
        print(f" Training Probabilistic model with {len(X)} samples...")
        
        # Group samples by reference point
        unique_coords = np.unique(y, axis=0)
        
        for coord in unique_coords:
            # Find all samples for this reference point
            mask = (y == coord).all(axis=1)
            samples = X[mask]
            
            if len(samples) > 0:
                # Calculate mean and covariance
                mean = np.mean(samples, axis=0)
                cov = self._estimate_covariance(samples)
                
                # Store distribution parameters
                self.reference_points[tuple(coord)] = {
                    'mean': mean,
                    'cov': cov,
                    'coord': coord,
                    'n_samples': len(samples)
                }
            else:
                print(f" No samples found for coordinate {coord}")
        
        print(f" Learned distributions for {len(self.reference_points)} reference points")
        
    def predict(self, X):
        """
        Predict locations using maximum likelihood estimation
        
        Args:
            X (np.array): Test features (M x D)
            
        Returns:
            np.array: Predicted coordinates (M x 2)
        """
        if not self.reference_points:
            raise ValueError("Model must be fitted before prediction")
            
        print(f" Predicting locations for {len(X)} test samples...")
        
        predictions = []
        
        for x_test in X:
            max_likelihood = -np.inf
            best_coord = None
            
            # Calculate likelihood for each reference point
            for coord_tuple, ref_data in self.reference_points.items():
                try:
                    # Calculate log-likelihood
                    likelihood = multivariate_normal.logpdf(
                        x_test, 
                        ref_data['mean'], 
                        ref_data['cov']
                    )
                    
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood
                        best_coord = ref_data['coord']
                        
                except Exception as e:
                    # Skip problematic distributions
                    continue
            
            if best_coord is not None:
                predictions.append(best_coord)
            else:
                # Fallback to origin if no valid prediction
                predictions.append([0, 0])
                
        predictions = np.array(predictions)
        print(f" Prediction complete")
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probability distribution over reference points
        
        Args:
            X (np.array): Test features
            
        Returns:
            tuple: (coordinates, probabilities) for each test sample
        """
        if not self.reference_points:
            raise ValueError("Model must be fitted before prediction")
            
        print(f" Computing probability distributions for {len(X)} test samples...")
        
        # Get reference coordinates in consistent order
        coord_keys = list(self.reference_points.keys())
        coordinates = np.array([self.reference_points[key]['coord'] for key in coord_keys])
        
        all_probabilities = []
        
        for x_test in X:
            log_likelihoods = []
            
            # Calculate log-likelihood for each reference point
            for coord_key in coord_keys:
                ref_data = self.reference_points[coord_key]
                try:
                    log_likelihood = multivariate_normal.logpdf(
                        x_test, 
                        ref_data['mean'], 
                        ref_data['cov']
                    )
                    log_likelihoods.append(log_likelihood)
                except:
                    log_likelihoods.append(-np.inf)
            
            log_likelihoods = np.array(log_likelihoods)
            
            # Convert to probabilities using softmax (log-sum-exp trick)
            max_log_likelihood = np.max(log_likelihoods)
            exp_likelihoods = np.exp(log_likelihoods - max_log_likelihood)
            probabilities = exp_likelihoods / np.sum(exp_likelihoods)
            
            all_probabilities.append(probabilities)
        
        all_probabilities = np.array(all_probabilities)
        
        print(f" Probability computation complete")
        
        return coordinates, all_probabilities
    
    def analyze_distributions(self):
        """
        Analyze the learned distributions
        
        Returns:
            dict: Analysis of the learned distributions
        """
        if not self.reference_points:
            raise ValueError("Model must be fitted before analysis")
        
        analysis = {
            'n_reference_points': len(self.reference_points),
            'reference_coordinates': [],
            'sample_counts': [],
            'mean_vectors': [],
            'covariance_determinants': [],
            'covariance_condition_numbers': []
        }
        
        for coord_tuple, ref_data in self.reference_points.items():
            analysis['reference_coordinates'].append(ref_data['coord'])
            analysis['sample_counts'].append(ref_data['n_samples'])
            analysis['mean_vectors'].append(ref_data['mean'])
            
            # Covariance analysis
            cov = ref_data['cov']
            det = np.linalg.det(cov)
            cond_num = np.linalg.cond(cov)
            
            analysis['covariance_determinants'].append(det)
            analysis['covariance_condition_numbers'].append(cond_num)
        
        # Summary statistics
        analysis['summary'] = {
            'avg_samples_per_point': np.mean(analysis['sample_counts']),
            'min_samples_per_point': np.min(analysis['sample_counts']),
            'max_samples_per_point': np.max(analysis['sample_counts']),
            'avg_covariance_det': np.mean(analysis['covariance_determinants']),
            'avg_condition_number': np.mean(analysis['covariance_condition_numbers']),
            'max_condition_number': np.max(analysis['covariance_condition_numbers'])
        }
        
        return analysis

class GaussianMixtureLocalizer:
    """GaussianMixtureLocalizer."""

    def __init__(self, n_components=2, covariance_type='full', regularization=1e-6):
        """
        Initialize GMM localizer
        
        Args:
            n_components (int): Number of Gaussian components per reference point
            covariance_type (str): Type of covariance ('full', 'tied', 'diag', 'spherical')
            regularization (float): Regularization for covariance
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.regularization = regularization
        self.reference_points = {}
        
        print(f" Initializing GMM Localizer (components={n_components}, cov_type={covariance_type})")
        
    def fit(self, X, y):
        """
        Learn GMM for each reference point
        
        Args:
            X (np.array): Training features
            y (np.array): Training coordinates
        """
        print(f" Training GMM model with {len(X)} samples...")
        
        # Group samples by reference point
        unique_coords = np.unique(y, axis=0)
        
        for coord in unique_coords:
            # Find all samples for this reference point
            mask = (y == coord).all(axis=1)
            samples = X[mask]
            
            if len(samples) >= self.n_components:
                # Fit GMM with multiple components
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    reg_covar=self.regularization,
                    random_state=42
                )
                gmm.fit(samples)
                
                self.reference_points[tuple(coord)] = {
                    'gmm': gmm,
                    'coord': coord,
                    'n_samples': len(samples)
                }
            elif len(samples) > 0:
                # Fall back to single Gaussian
                gmm = GaussianMixture(
                    n_components=1,
                    covariance_type=self.covariance_type,
                    reg_covar=self.regularization,
                    random_state=42
                )
                gmm.fit(samples)
                
                self.reference_points[tuple(coord)] = {
                    'gmm': gmm,
                    'coord': coord,
                    'n_samples': len(samples)
                }
        
        print(f" Learned GMMs for {len(self.reference_points)} reference points")
        
    def predict(self, X):
        """
        Predict locations using GMM likelihood
        
        Args:
            X (np.array): Test features
            
        Returns:
            np.array: Predicted coordinates
        """
        if not self.reference_points:
            raise ValueError("Model must be fitted before prediction")
            
        print(f" Predicting locations for {len(X)} test samples...")
        
        predictions = []
        
        for x_test in X:
            max_likelihood = -np.inf
            best_coord = None
            
            # Calculate likelihood for each reference point GMM
            for coord_tuple, ref_data in self.reference_points.items():
                try:
                    # Calculate log-likelihood using GMM
                    likelihood = ref_data['gmm'].score_samples([x_test])[0]
                    
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood
                        best_coord = ref_data['coord']
                        
                except Exception as e:
                    continue
            
            if best_coord is not None:
                predictions.append(best_coord)
            else:
                predictions.append([0, 0])
                
        predictions = np.array(predictions)
        print(f" GMM prediction complete")
        
        return predictions

class BayesianLocalizer:
    """BayesianLocalizer."""

    def __init__(self, prior_type='uniform', smoothing=1e-6):
        """
        Initialize Bayesian localizer
        
        Args:
            prior_type (str): Type of prior ('uniform', 'distance_based', 'density_based')
            smoothing (float): Smoothing parameter
        """
        self.prior_type = prior_type
        self.smoothing = smoothing
        self.reference_points = {}
        self.priors = {}
        
        print(f" Initializing Bayesian Localizer (prior={prior_type})")
        
    def _compute_priors(self, coordinates):
        """Compute prior probabilities for reference points"""
        
        n_points = len(coordinates)
        
        if self.prior_type == 'uniform':
            # Uniform prior
            prior_prob = 1.0 / n_points
            for coord in coordinates:
                self.priors[tuple(coord)] = prior_prob
                
        elif self.prior_type == 'distance_based':
            # Prior based on distance from center
            center = np.mean(coordinates, axis=0)
            distances = np.linalg.norm(coordinates - center, axis=1)
            
            # Closer to center = higher prior
            weights = 1.0 / (distances + 0.1)  # Avoid division by zero
            weights = weights / np.sum(weights)  # Normalize
            
            for i, coord in enumerate(coordinates):
                self.priors[tuple(coord)] = weights[i]
                
        elif self.prior_type == 'density_based':
            # Prior based on local density of reference points
            densities = []
            for coord in coordinates:
                # Count neighbors within radius
                distances = np.linalg.norm(coordinates - coord, axis=1)
                density = np.sum(distances < 2.0)  # Within 2m
                densities.append(density)
            
            densities = np.array(densities, dtype=float)
            densities = densities / np.sum(densities)  # Normalize
            
            for i, coord in enumerate(coordinates):
                self.priors[tuple(coord)] = densities[i]
        
    def fit(self, X, y):
        """
        Learn Bayesian model
        
        Args:
            X (np.array): Training features
            y (np.array): Training coordinates
        """
        print(f" Training Bayesian model with {len(X)} samples...")
        
        # First, fit basic probabilistic model
        base_model = ProbabilisticLocalizer(smoothing=self.smoothing)
        base_model.fit(X, y)
        self.reference_points = base_model.reference_points
        
        # Compute priors
        unique_coords = np.unique(y, axis=0)
        self._compute_priors(unique_coords)
        
        print(f" Bayesian model trained with {self.prior_type} priors")
        
    def predict(self, X):
        """
        Predict using Bayesian inference
        
        Args:
            X (np.array): Test features
            
        Returns:
            np.array: Predicted coordinates
        """
        if not self.reference_points:
            raise ValueError("Model must be fitted before prediction")
            
        print(f" Bayesian prediction for {len(X)} test samples...")
        
        predictions = []
        
        for x_test in X:
            max_posterior = -np.inf
            best_coord = None
            
            # Calculate posterior for each reference point
            for coord_tuple, ref_data in self.reference_points.items():
                try:
                    # Likelihood
                    likelihood = multivariate_normal.logpdf(
                        x_test, 
                        ref_data['mean'], 
                        ref_data['cov']
                    )
                    
                    # Prior
                    prior = np.log(self.priors[coord_tuple] + 1e-10)
                    
                    # Posterior = likelihood + prior (in log space)
                    posterior = likelihood + prior
                    
                    if posterior > max_posterior:
                        max_posterior = posterior
                        best_coord = ref_data['coord']
                        
                except Exception as e:
                    continue
            
            if best_coord is not None:
                predictions.append(best_coord)
            else:
                predictions.append([0, 0])
                
        predictions = np.array(predictions)
        print(f" Bayesian prediction complete")
        
        return predictions
#!/usr/bin/env python3
"""
IDW Pipeline.

pipeline for IDW localization including data loading,
preprocessing, training, evaluation, and visualization.
"""

import numpy as np
from pathlib import Path
import sys
import time

# Add shared modules to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'shared'))
from data_loader import load_amplitude_phase_data, create_train_test_split, load_test_data
from preprocessing import prepare_classical_features
from evaluation import calculate_localization_metrics, plot_error_distribution, plot_predictions_scatter, save_results

# Import model
from .model import IDWLocalizer, AdaptiveIDWLocalizer, MultiPowerIDWEnsemble

class IDWPipeline:
    """IDWPipeline."""

    def __init__(self, output_dir="idw_results"):
        """
        Initialize IDW pipeline
        
        Args:
            output_dir (str): Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.preprocessing_info = None
        
        print(" IDW Pipeline Initialized")
        print(f" Output directory: {self.output_dir}")
        
    def load_and_preprocess_data(self, data_source="csi_dataset", include_statistical=False, 
                                scaler_type='standard', use_test_split=True):
        """
        Load and preprocess data for IDW models
        
        Args:
            data_source (str): Data source type
            include_statistical (bool): Whether to add statistical features
            scaler_type (str): Type of scaling
            use_test_split (bool): Whether to use train/test split or load separate test data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        
        print(" Loading and preprocessing data for IDW...")
        
        # Load training data
        X, y, coordinates = load_amplitude_phase_data(data_source=data_source)
        
        if use_test_split:
            # Create train/test split
            X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        else:
            # Use all data for training, load separate test set
            X_train, y_train = X, y
            X_test, y_test = load_test_data()
        
        # Preprocess features
        prep_result = prepare_classical_features(
            X_train, X_test, 
            include_statistical=include_statistical,
            scaler_type=scaler_type,
            remove_outliers_flag=False
        )
        
        self.preprocessing_info = prep_result['feature_info']
        
        return prep_result['X_train'], prep_result['X_test'], y_train, y_test
    
    def run_basic_idw_evaluation(self, X_train, X_test, y_train, y_test, 
                                power_values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]):
        """
        Evaluate basic IDW models with different power values
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            power_values (list): List of power values to evaluate
            
        Returns:
            list: Results for all power values
        """
        
        print(f" Evaluating basic IDW models...")
        print(f"   Power values: {power_values}")
        
        all_results = []
        
        for power in power_values:
            print(f"\n--- Evaluating IDW with power={power} ---")
            
            # Create and train model
            model = IDWLocalizer(power=power)
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, f"IDW (p={power})")
            
            # Add timing and parameter information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['power_value'] = power
            
            # Store model and results
            self.models[f"idw_p{power}"] = model
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, f"IDW (p={power})", self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, f"idw_p{power}")
        
        return all_results
    
    def run_distance_metric_evaluation(self, X_train, X_test, y_train, y_test, 
                                     metrics=['euclidean', 'manhattan'], power=2.0):
        """
        Evaluate IDW models with different distance metrics
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            metrics (list): List of distance metrics to evaluate
            power (float): Power parameter for all models
            
        Returns:
            list: Results for different distance metrics
        """
        
        print(f" Evaluating IDW with different distance metrics...")
        print(f"   Metrics: {metrics}, Power: {power}")
        
        all_results = []
        
        for metric in metrics:
            print(f"\n--- Evaluating IDW with {metric} distance ---")
            
            # Create and train model
            model = IDWLocalizer(power=power, distance_metric=metric)
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, f"IDW {metric} (p={power})")
            
            # Add information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['power_value'] = power
            results['distance_metric'] = metric
            
            # Store model and results
            self.models[f"idw_{metric}_p{power}"] = model
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, f"IDW {metric} (p={power})", 
                                   self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, f"idw_{metric}_p{power}")
        
        return all_results
    
    def run_adaptive_idw_evaluation(self, X_train, X_test, y_train, y_test):
        """
        Evaluate adaptive IDW models
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            
        Returns:
            list: Results for adaptive IDW models
        """
        
        print(f" Evaluating adaptive IDW models...")
        
        adaptation_methods = ['density', 'variance', 'distance']
        all_results = []
        
        for method in adaptation_methods:
            print(f"\n--- Evaluating Adaptive IDW ({method}) ---")
            
            # Create and train model
            model = AdaptiveIDWLocalizer(base_power=2.0, adaptation_method=method)
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, f"Adaptive IDW ({method})")
            
            # Add information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['adaptation_method'] = method
            results['is_adaptive'] = True
            
            # Store model and results
            self.models[f"adaptive_idw_{method}"] = model
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, f"Adaptive IDW ({method})", 
                                   self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, f"adaptive_idw_{method}")
        
        return all_results
    
    def run_ensemble_evaluation(self, X_train, X_test, y_train, y_test):
        """
        Evaluate IDW ensemble models
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            
        Returns:
            list: Results for ensemble models
        """
        
        print(f" Evaluating IDW ensemble models...")
        
        ensemble_configs = [
            {'powers': [1.0, 2.0, 3.0], 'method': 'average', 'name': 'Ensemble p123 (avg)'},
            {'powers': [0.5, 1.5, 2.5], 'method': 'average', 'name': 'Ensemble p0.5-1.5-2.5 (avg)'},
            {'powers': [1.0, 2.0, 3.0, 4.0], 'method': 'average', 'name': 'Ensemble p1234 (avg)'},
            {'powers': [1.0, 2.0, 3.0], 'method': 'weighted_average', 'name': 'Ensemble p123 (weighted)'},
            {'powers': [0.5, 1.5, 2.5, 3.5], 'method': 'median', 'name': 'Ensemble varied (median)'}
        ]
        
        all_results = []
        
        for config in ensemble_configs:
            print(f"\n--- Evaluating {config['name']} ---")
            
            # Create and train ensemble
            ensemble = MultiPowerIDWEnsemble(
                power_values=config['powers'],
                ensemble_method=config['method']
            )
            
            start_time = time.time()
            ensemble.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = ensemble.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, config['name'])
            
            # Add ensemble information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['power_values'] = config['powers']
            results['ensemble_method'] = config['method']
            results['is_ensemble'] = True
            
            # Store model and results
            model_key = config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
            self.models[model_key] = ensemble
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, config['name'], self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, model_key)
        
        return all_results
    
    def run_neighbor_limit_evaluation(self, X_train, X_test, y_train, y_test, 
                                    max_neighbors_list=[10, 20, 50, 100], power=2.0):
        """
        Evaluate IDW with limited number of neighbors
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            max_neighbors_list (list): List of neighbor limits to test
            power (float): Power parameter
            
        Returns:
            list: Results for different neighbor limits
        """
        
        print(f" Evaluating IDW with neighbor limits...")
        print(f"   Neighbor limits: {max_neighbors_list}, Power: {power}")
        
        all_results = []
        
        for max_neighbors in max_neighbors_list:
            print(f"\n--- Evaluating IDW with max {max_neighbors} neighbors ---")
            
            # Create and train model
            model = IDWLocalizer(power=power, max_neighbors=max_neighbors)
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, f"IDW-{max_neighbors}nn (p={power})")
            
            # Add information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['power_value'] = power
            results['max_neighbors'] = max_neighbors
            
            # Store model and results
            self.models[f"idw_{max_neighbors}nn_p{power}"] = model
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, f"IDW-{max_neighbors}nn (p={power})", 
                                   self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, f"idw_{max_neighbors}nn_p{power}")
        
        return all_results
    
    def run_complete_evaluation(self, data_source="csi_dataset", include_statistical=False):
        """
        Run complete IDW evaluation pipeline
        
        Args:
            data_source (str): Data source type
            include_statistical (bool): Whether to include statistical features
            
        Returns:
            dict: results summary
        """
        
        print(" Running IDW Evaluation Pipeline")
        print("=" * 50)
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data(
            data_source=data_source,
            include_statistical=include_statistical,
            use_test_split=False  # Use separate test set
        )
        
        # Store data info
        data_info = {
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'data_source': data_source,
            'statistical_features': include_statistical,
            'preprocessing_info': self.preprocessing_info
        }
        
        print(f" Data loaded: {data_info['n_train_samples']} train, {data_info['n_test_samples']} test")
        print(f" Features: {data_info['n_features']} (statistical: {include_statistical})")
        
        # 1. Basic IDW models with different powers
        print(f"\n1⃣ Basic IDW Models")
        basic_results = self.run_basic_idw_evaluation(X_train, X_test, y_train, y_test)
        
        # 2. Different distance metrics
        print(f"\n2⃣ Distance Metric Comparison")
        metric_results = self.run_distance_metric_evaluation(X_train, X_test, y_train, y_test)
        
        # 3. Adaptive IDW models
        print(f"\n3⃣ Adaptive IDW Models")
        adaptive_results = self.run_adaptive_idw_evaluation(X_train, X_test, y_train, y_test)
        
        # 4. Ensemble IDW models
        print(f"\n4⃣ Ensemble IDW Models")
        ensemble_results = self.run_ensemble_evaluation(X_train, X_test, y_train, y_test)
        
        # 5. Neighbor-limited IDW models
        print(f"\n5⃣ Neighbor-Limited IDW Models")
        neighbor_results = self.run_neighbor_limit_evaluation(X_train, X_test, y_train, y_test)
        
        # Combine all results
        all_results = basic_results + metric_results + adaptive_results + ensemble_results + neighbor_results
        
        # Create comparison visualization
        from evaluation import compare_models_cdf, create_performance_summary_table
        compare_models_cdf(all_results, self.output_dir, show_plot=False)
        create_performance_summary_table(all_results, self.output_dir)
        
        # Prepare final summary
        summary = {
            'data_info': data_info,
            'basic_results': basic_results,
            'metric_results': metric_results,
            'adaptive_results': adaptive_results,
            'ensemble_results': ensemble_results,
            'neighbor_results': neighbor_results,
            'all_results': all_results,
            'best_model': min(all_results, key=lambda x: x['median_error']),
            'models_trained': len(all_results)
        }
        
        # Save complete summary
        import json
        summary_path = self.output_dir / "idw_complete_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n IDW Evaluation Complete!")
        print(f" Evaluated {summary['models_trained']} models")
        print(f" Best model: {summary['best_model']['model']} (median: {summary['best_model']['median_error']:.3f}m)")
        print(f" Results saved to: {self.output_dir}")
        
        return summary

def main():
    """Main function to run IDW pipeline"""
    
    pipeline = IDWPipeline()
    
    # Run complete evaluation
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=False
    )
    
    print("\n--- IDW Pipeline Summary ---")
    print(f"Best performing model: {summary['best_model']['model']}")
    print(f"Median error: {summary['best_model']['median_error']:.3f}m")
    print(f"1m accuracy: {summary['best_model']['accuracy_1m']:.1f}%")
    print(f"2m accuracy: {summary['best_model']['accuracy_2m']:.1f}%")

if __name__ == "__main__":
    main()

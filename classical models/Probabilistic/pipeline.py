#!/usr/bin/env python3
"""
Probabilistic Pipeline

pipeline for probabilistic localization including data loading,
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
from .model import ProbabilisticLocalizer, GaussianMixtureLocalizer, BayesianLocalizer

class ProbabilisticPipeline:
    """ProbabilisticPipeline."""

    def __init__(self, output_dir="probabilistic_results"):
        """
        Initialize Probabilistic pipeline
        
        Args:
            output_dir (str): Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.preprocessing_info = None
        
        print(" Probabilistic Pipeline Initialized")
        print(f" Output directory: {self.output_dir}")
        
    def load_and_preprocess_data(self, data_source="csi_dataset", include_statistical=False, 
                                scaler_type='standard', use_test_split=True):
        """
        Load and preprocess data for probabilistic models
        
        Args:
            data_source (str): Data source type
            include_statistical (bool): Whether to add statistical features
            scaler_type (str): Type of scaling
            use_test_split (bool): Whether to use train/test split or load separate test data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        
        print(" Loading and preprocessing data for Probabilistic models...")
        
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
    
    def run_basic_probabilistic_evaluation(self, X_train, X_test, y_train, y_test):
        """
        Evaluate basic probabilistic models with different covariance estimators
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            
        Returns:
            list: Results for different covariance types
        """
        
        print(f" Evaluating basic probabilistic models...")
        
        covariance_configs = [
            {'type': 'empirical', 'smoothing': 1e-6, 'name': 'Probabilistic (empirical)'},
            {'type': 'ledoit_wolf', 'smoothing': 1e-6, 'name': 'Probabilistic (Ledoit-Wolf)'},
            {'type': 'oas', 'smoothing': 1e-6, 'name': 'Probabilistic (OAS)'},
            {'type': 'shrunk', 'smoothing': 1e-6, 'name': 'Probabilistic (shrunk)'},
            {'type': 'empirical', 'smoothing': 1e-4, 'name': 'Probabilistic (high smoothing)'},
            {'type': 'empirical', 'smoothing': 1e-8, 'name': 'Probabilistic (low smoothing)'}
        ]
        
        all_results = []
        
        for config in covariance_configs:
            print(f"\n--- Evaluating {config['name']} ---")
            
            # Create and train model
            model = ProbabilisticLocalizer(
                covariance_type=config['type'],
                smoothing=config['smoothing']
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, config['name'])
            
            # Add timing and configuration information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['covariance_type'] = config['type']
            results['smoothing'] = config['smoothing']
            
            # Analyze distributions
            try:
                distribution_analysis = model.analyze_distributions()
                results['distribution_analysis'] = distribution_analysis['summary']
            except Exception as e:
                print(f"    Could not analyze distributions: {e}")
                results['distribution_analysis'] = None
            
            # Store model and results
            model_key = config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            self.models[model_key] = model
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, config['name'], self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, model_key)
        
        return all_results
    
    def run_gmm_evaluation(self, X_train, X_test, y_train, y_test):
        """
        Evaluate Gaussian Mixture Model localization
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            
        Returns:
            list: Results for GMM models
        """
        
        print(f" Evaluating Gaussian Mixture Model localization...")
        
        gmm_configs = [
            {'n_components': 1, 'cov_type': 'full', 'name': 'GMM-1 (full)'},
            {'n_components': 2, 'cov_type': 'full', 'name': 'GMM-2 (full)'},
            {'n_components': 3, 'cov_type': 'full', 'name': 'GMM-3 (full)'},
            {'n_components': 2, 'cov_type': 'diag', 'name': 'GMM-2 (diag)'},
            {'n_components': 2, 'cov_type': 'tied', 'name': 'GMM-2 (tied)'},
            {'n_components': 2, 'cov_type': 'spherical', 'name': 'GMM-2 (spherical)'}
        ]
        
        all_results = []
        
        for config in gmm_configs:
            print(f"\n--- Evaluating {config['name']} ---")
            
            # Create and train model
            model = GaussianMixtureLocalizer(
                n_components=config['n_components'],
                covariance_type=config['cov_type']
            )
            
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Make predictions
                start_time = time.time()
                y_pred = model.predict(X_test)
                pred_time = time.time() - start_time
                
                # Calculate metrics
                results = calculate_localization_metrics(y_test, y_pred, config['name'])
                
                # Add information
                results['train_time'] = train_time
                results['prediction_time'] = pred_time
                results['n_components'] = config['n_components']
                results['covariance_type'] = config['cov_type']
                results['model_type'] = 'GMM'
                
                # Store model and results
                model_key = config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                self.models[model_key] = model
                all_results.append(results)
                
                # Create visualizations
                plot_error_distribution(results, self.output_dir, show_plot=False)
                plot_predictions_scatter(y_test, y_pred, config['name'], self.output_dir, show_plot=False)
                
                # Save individual results
                save_results(results, self.output_dir, model_key)
                
            except Exception as e:
                print(f"    Error with {config['name']}: {e}")
                continue
        
        return all_results
    
    def run_bayesian_evaluation(self, X_train, X_test, y_train, y_test):
        """
        Evaluate Bayesian probabilistic models
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            
        Returns:
            list: Results for Bayesian models
        """
        
        print(f" Evaluating Bayesian probabilistic models...")
        
        bayesian_configs = [
            {'prior_type': 'uniform', 'name': 'Bayesian (uniform prior)'},
            {'prior_type': 'distance_based', 'name': 'Bayesian (distance prior)'},
            {'prior_type': 'density_based', 'name': 'Bayesian (density prior)'}
        ]
        
        all_results = []
        
        for config in bayesian_configs:
            print(f"\n--- Evaluating {config['name']} ---")
            
            # Create and train model
            model = BayesianLocalizer(prior_type=config['prior_type'])
            
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Make predictions
                start_time = time.time()
                y_pred = model.predict(X_test)
                pred_time = time.time() - start_time
                
                # Calculate metrics
                results = calculate_localization_metrics(y_test, y_pred, config['name'])
                
                # Add information
                results['train_time'] = train_time
                results['prediction_time'] = pred_time
                results['prior_type'] = config['prior_type']
                results['model_type'] = 'Bayesian'
                
                # Store model and results
                model_key = config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
                self.models[model_key] = model
                all_results.append(results)
                
                # Create visualizations
                plot_error_distribution(results, self.output_dir, show_plot=False)
                plot_predictions_scatter(y_test, y_pred, config['name'], self.output_dir, show_plot=False)
                
                # Save individual results
                save_results(results, self.output_dir, model_key)
                
            except Exception as e:
                print(f"    Error with {config['name']}: {e}")
                continue
        
        return all_results
    
    def run_regularization_analysis(self, X_train, X_test, y_train, y_test):
        """
        Analyze effect of different regularization parameters
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            
        Returns:
            list: Results for different regularization values
        """
        
        print(f" Analyzing regularization effects...")
        
        smoothing_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        all_results = []
        
        for smoothing in smoothing_values:
            print(f"\n--- Evaluating smoothing={smoothing:.0e} ---")
            
            # Create and train model
            model = ProbabilisticLocalizer(
                covariance_type='empirical',
                smoothing=smoothing
            )
            
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Make predictions
                start_time = time.time()
                y_pred = model.predict(X_test)
                pred_time = time.time() - start_time
                
                # Calculate metrics
                results = calculate_localization_metrics(y_test, y_pred, f"Prob (smooth={smoothing:.0e})")
                
                # Add information
                results['train_time'] = train_time
                results['prediction_time'] = pred_time
                results['smoothing'] = smoothing
                results['analysis_type'] = 'regularization'
                
                all_results.append(results)
                
            except Exception as e:
                print(f"    Error with smoothing={smoothing}: {e}")
                continue
        
        return all_results
    
    def run_complete_evaluation(self, data_source="csi_dataset", include_statistical=False):
        """
        Run complete probabilistic evaluation pipeline
        
        Args:
            data_source (str): Data source type
            include_statistical (bool): Whether to include statistical features
            
        Returns:
            dict: results summary
        """
        
        print(" Running Probabilistic Evaluation Pipeline")
        print("=" * 60)
        
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
        
        # 1. Basic probabilistic models
        print(f"\n1⃣ Basic Probabilistic Models")
        basic_results = self.run_basic_probabilistic_evaluation(X_train, X_test, y_train, y_test)
        
        # 2. Gaussian Mixture Models
        print(f"\n2⃣ Gaussian Mixture Models")
        gmm_results = self.run_gmm_evaluation(X_train, X_test, y_train, y_test)
        
        # 3. Bayesian models
        print(f"\n3⃣ Bayesian Models")
        bayesian_results = self.run_bayesian_evaluation(X_train, X_test, y_train, y_test)
        
        # 4. Regularization analysis
        print(f"\n4⃣ Regularization Analysis")
        regularization_results = self.run_regularization_analysis(X_train, X_test, y_train, y_test)
        
        # Combine all results
        all_results = basic_results + gmm_results + bayesian_results + regularization_results
        
        # Create comparison visualization
        from evaluation import compare_models_cdf, create_performance_summary_table
        compare_models_cdf(all_results, self.output_dir, show_plot=False)
        create_performance_summary_table(all_results, self.output_dir)
        
        # Prepare final summary
        summary = {
            'data_info': data_info,
            'basic_results': basic_results,
            'gmm_results': gmm_results,
            'bayesian_results': bayesian_results,
            'regularization_results': regularization_results,
            'all_results': all_results,
            'best_model': min(all_results, key=lambda x: x['median_error']),
            'models_trained': len(all_results)
        }
        
        # Analyze model type performance
        model_type_analysis = self._analyze_by_model_type(all_results)
        summary['model_type_analysis'] = model_type_analysis
        
        # Save complete summary
        import json
        summary_path = self.output_dir / "probabilistic_complete_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n Probabilistic Evaluation Complete!")
        print(f" Evaluated {summary['models_trained']} models")
        print(f" Best model: {summary['best_model']['model']} (median: {summary['best_model']['median_error']:.3f}m)")
        print(f" Results saved to: {self.output_dir}")
        
        return summary
    
    def _analyze_by_model_type(self, all_results):
        """Analyze performance by model type"""
        
        type_groups = {
            'Basic Probabilistic': [],
            'GMM': [],
            'Bayesian': [],
            'Regularization': []
        }
        
        for result in all_results:
            if 'model_type' in result:
                if result['model_type'] == 'GMM':
                    type_groups['GMM'].append(result)
                elif result['model_type'] == 'Bayesian':
                    type_groups['Bayesian'].append(result)
            elif 'analysis_type' in result and result['analysis_type'] == 'regularization':
                type_groups['Regularization'].append(result)
            else:
                type_groups['Basic Probabilistic'].append(result)
        
        analysis = {}
        for type_name, results in type_groups.items():
            if results:
                median_errors = [r['median_error'] for r in results]
                analysis[type_name] = {
                    'count': len(results),
                    'best_median': min(median_errors),
                    'worst_median': max(median_errors),
                    'avg_median': np.mean(median_errors),
                    'best_model': min(results, key=lambda x: x['median_error'])['model']
                }
        
        return analysis

def main():
    """Main function to run Probabilistic pipeline"""
    
    pipeline = ProbabilisticPipeline()
    
    # Run complete evaluation
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=False
    )
    
    print("\n--- Probabilistic Pipeline Summary ---")
    print(f"Best performing model: {summary['best_model']['model']}")
    print(f"Median error: {summary['best_model']['median_error']:.3f}m")
    print(f"1m accuracy: {summary['best_model']['accuracy_1m']:.1f}%")
    print(f"2m accuracy: {summary['best_model']['accuracy_2m']:.1f}%")

if __name__ == "__main__":
    main()
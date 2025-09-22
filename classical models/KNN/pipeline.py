#!/usr/bin/env python3
"""
k-NN Pipeline
=============

Complete pipeline for k-NN localization including data loading,
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
from .model import KNNLocalizer, MultiKNNEnsemble

class KNNPipeline:
    """Complete pipeline for k-NN localization"""
    
    def __init__(self, output_dir="knn_results"):
        """
        Initialize k-NN pipeline
        
        Args:
            output_dir (str): Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.preprocessing_info = None
        
        print("🎯 k-NN Pipeline Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        
    def load_and_preprocess_data(self, data_source="csi_dataset", include_statistical=False, 
                                scaler_type='standard', use_test_split=True):
        """
        Load and preprocess data for k-NN models
        
        Args:
            data_source (str): Data source type
            include_statistical (bool): Whether to add statistical features
            scaler_type (str): Type of scaling
            use_test_split (bool): Whether to use train/test split or load separate test data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        
        print("📂 Loading and preprocessing data for k-NN...")
        
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
    
    def run_single_knn_evaluation(self, X_train, X_test, y_train, y_test, k_values=[1, 3, 5, 7, 9]):
        """
        Evaluate individual k-NN models with different k values
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            k_values (list): List of k values to evaluate
            
        Returns:
            list: Results for all k values
        """
        
        print(f"🔬 Evaluating individual k-NN models...")
        print(f"   k values: {k_values}")
        
        all_results = []
        
        for k in k_values:
            print(f"\n--- Evaluating k-NN with k={k} ---")
            
            # Create and train model
            model = KNNLocalizer(k=k, weights='uniform')
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, f"k-NN (k={k})")
            
            # Add timing information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['k_value'] = k
            
            # Store model and results
            self.models[f"knn_k{k}"] = model
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, f"k-NN (k={k})", self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, f"knn_k{k}")
        
        return all_results
    
    def run_distance_weighted_knn(self, X_train, X_test, y_train, y_test, k_values=[3, 5, 7]):
        """
        Evaluate distance-weighted k-NN models
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            k_values (list): List of k values for weighted k-NN
            
        Returns:
            list: Results for weighted k-NN models
        """
        
        print(f"🔬 Evaluating distance-weighted k-NN models...")
        print(f"   k values: {k_values}")
        
        all_results = []
        
        for k in k_values:
            print(f"\n--- Evaluating Weighted k-NN with k={k} ---")
            
            # Create and train model with distance weighting
            model = KNNLocalizer(k=k, weights='distance')
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            results = calculate_localization_metrics(y_test, y_pred, f"Weighted k-NN (k={k})")
            
            # Add timing information
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['k_value'] = k
            results['weighted'] = True
            
            # Store model and results
            self.models[f"weighted_knn_k{k}"] = model
            all_results.append(results)
            
            # Create visualizations
            plot_error_distribution(results, self.output_dir, show_plot=False)
            plot_predictions_scatter(y_test, y_pred, f"Weighted k-NN (k={k})", self.output_dir, show_plot=False)
            
            # Save individual results
            save_results(results, self.output_dir, f"weighted_knn_k{k}")
        
        return all_results
    
    def run_ensemble_evaluation(self, X_train, X_test, y_train, y_test):
        """
        Evaluate k-NN ensemble models
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            
        Returns:
            list: Results for ensemble models
        """
        
        print(f"🔬 Evaluating k-NN ensemble models...")
        
        ensemble_configs = [
            {'k_values': [1, 3, 5], 'method': 'average', 'name': 'Ensemble k135 (avg)'},
            {'k_values': [3, 5, 7], 'method': 'average', 'name': 'Ensemble k357 (avg)'},
            {'k_values': [1, 3, 5, 7, 9], 'method': 'average', 'name': 'Ensemble k13579 (avg)'},
            {'k_values': [1, 3, 5], 'method': 'weighted_average', 'name': 'Ensemble k135 (weighted)'},
            {'k_values': [3, 5, 7], 'method': 'median', 'name': 'Ensemble k357 (median)'}
        ]
        
        all_results = []
        
        for config in ensemble_configs:
            print(f"\n--- Evaluating {config['name']} ---")
            
            # Create and train ensemble
            ensemble = MultiKNNEnsemble(
                k_values=config['k_values'],
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
            results['k_values'] = config['k_values']
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
    
    def run_complete_evaluation(self, data_source="csi_dataset", include_statistical=False):
        """
        Run complete k-NN evaluation pipeline
        
        Args:
            data_source (str): Data source type
            include_statistical (bool): Whether to include statistical features
            
        Returns:
            dict: Complete results summary
        """
        
        print("🚀 Running Complete k-NN Evaluation Pipeline")
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
        
        print(f"📊 Data loaded: {data_info['n_train_samples']} train, {data_info['n_test_samples']} test")
        print(f"🔧 Features: {data_info['n_features']} (statistical: {include_statistical})")
        
        # 1. Individual k-NN models
        print(f"\n1️⃣ Individual k-NN Models")
        individual_results = self.run_single_knn_evaluation(X_train, X_test, y_train, y_test)
        
        # 2. Distance-weighted k-NN models
        print(f"\n2️⃣ Distance-Weighted k-NN Models")
        weighted_results = self.run_distance_weighted_knn(X_train, X_test, y_train, y_test)
        
        # 3. Ensemble k-NN models
        print(f"\n3️⃣ Ensemble k-NN Models")
        ensemble_results = self.run_ensemble_evaluation(X_train, X_test, y_train, y_test)
        
        # Combine all results
        all_results = individual_results + weighted_results + ensemble_results
        
        # Create comparison visualization
        from evaluation import compare_models_cdf, create_performance_summary_table
        compare_models_cdf(all_results, self.output_dir, show_plot=False)
        create_performance_summary_table(all_results, self.output_dir)
        
        # Prepare final summary
        summary = {
            'data_info': data_info,
            'individual_results': individual_results,
            'weighted_results': weighted_results,
            'ensemble_results': ensemble_results,
            'all_results': all_results,
            'best_model': min(all_results, key=lambda x: x['median_error']),
            'models_trained': len(all_results)
        }
        
        # Save complete summary
        import json
        summary_path = self.output_dir / "knn_complete_summary.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if key in ['individual_results', 'weighted_results', 'ensemble_results', 'all_results']:
                json_summary[key] = value  # These are already JSON-serializable
            else:
                json_summary[key] = value
        
        with open(summary_path, 'w') as f:
            json.dump(json_summary, f, indent=4)
        
        print(f"\n🎉 k-NN Evaluation Complete!")
        print(f"✅ Evaluated {summary['models_trained']} models")
        print(f"🏆 Best model: {summary['best_model']['model']} (median: {summary['best_model']['median_error']:.3f}m)")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return summary

def main():
    """Main function to run k-NN pipeline"""
    
    pipeline = KNNPipeline()
    
    # Run complete evaluation
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=False
    )
    
    print("\n--- k-NN Pipeline Summary ---")
    print(f"Best performing model: {summary['best_model']['model']}")
    print(f"Median error: {summary['best_model']['median_error']:.3f}m")
    print(f"1m accuracy: {summary['best_model']['accuracy_1m']:.1f}%")
    print(f"2m accuracy: {summary['best_model']['accuracy_2m']:.1f}%")

if __name__ == "__main__":
    main()

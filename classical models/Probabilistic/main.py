#!/usr/bin/env python3
"""
Probabilistic Main.

Main script to run probabilistic localization evaluation.
Provides command-line interface for different evaluation modes.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from pipeline import ProbabilisticPipeline

def run_basic_evaluation():
    """Run basic probabilistic evaluation with default parameters"""
    
    print(" Running Basic Probabilistic Evaluation")
    print("=" * 45)
    
    pipeline = ProbabilisticPipeline(output_dir="probabilistic_basic_results")
    
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=False
    )
    
    return summary

def run_enhanced_evaluation():
    """Run enhanced probabilistic evaluation with statistical features"""
    
    print(" Running Probabilistic Evaluation")
    print("=" * 45)
    
    pipeline = ProbabilisticPipeline(output_dir="probabilistic_enhanced_results")
    
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=True
    )
    
    return summary

def run_covariance_analysis():
    """Run focused covariance estimation analysis"""
    
    print(" Running Probabilistic Covariance Analysis")
    print("=" * 45)
    
    pipeline = ProbabilisticPipeline(output_dir="probabilistic_covariance_analysis")
    
    # Load data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Test different covariance estimators
    results = pipeline.run_basic_probabilistic_evaluation(X_train, X_test, y_train, y_test)
    
    # Analyze covariance estimator performance
    print(f"\n Covariance Estimator Analysis:")
    print(f"{'Estimator':<20} {'Median Error (m)':<15} {'1m Accuracy (%)':<15} {'2m Accuracy (%)':<15}")
    print("-" * 75)
    
    for result in results:
        cov_type = result.get('covariance_type', 'unknown')
        smoothing = result.get('smoothing', 0)
        estimator_name = f"{cov_type} (s={smoothing:.0e})"
        
        median_err = result['median_error']
        acc_1m = result['accuracy_1m']
        acc_2m = result['accuracy_2m']
        print(f"{estimator_name:<20} {median_err:<15.3f} {acc_1m:<15.1f} {acc_2m:<15.1f}")
    
    # Find best covariance estimator
    best_result = min(results, key=lambda x: x['median_error'])
    print(f"\n Best Covariance Estimator:")
    print(f"   Type: {best_result.get('covariance_type', 'unknown')}")
    print(f"   Smoothing: {best_result.get('smoothing', 0):.0e}")
    print(f"   Median Error: {best_result['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_result['accuracy_1m']:.1f}%")
    
    return results

def run_gmm_analysis():
    """Run Gaussian Mixture Model analysis"""
    
    print(" Running GMM Analysis")
    print("=" * 25)
    
    pipeline = ProbabilisticPipeline(output_dir="probabilistic_gmm_analysis")
    
    # Load data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Test GMM models
    results = pipeline.run_gmm_evaluation(X_train, X_test, y_train, y_test)
    
    if results:
        # Analyze GMM performance
        print(f"\n GMM Configuration Analysis:")
        print(f"{'Configuration':<20} {'Median Error (m)':<15} {'1m Accuracy (%)':<15} {'2m Accuracy (%)':<15}")
        print("-" * 75)
        
        for result in results:
            n_comp = result.get('n_components', '?')
            cov_type = result.get('covariance_type', 'unknown')
            config_name = f"GMM-{n_comp} ({cov_type})"
            
            median_err = result['median_error']
            acc_1m = result['accuracy_1m']
            acc_2m = result['accuracy_2m']
            print(f"{config_name:<20} {median_err:<15.3f} {acc_1m:<15.1f} {acc_2m:<15.1f}")
        
        # Find best GMM configuration
        best_result = min(results, key=lambda x: x['median_error'])
        print(f"\n Best GMM Configuration:")
        print(f"   Components: {best_result.get('n_components', '?')}")
        print(f"   Covariance: {best_result.get('covariance_type', 'unknown')}")
        print(f"   Median Error: {best_result['median_error']:.3f}m")
        print(f"   1m Accuracy: {best_result['accuracy_1m']:.1f}%")
    else:
        print(" No successful GMM evaluations")
    
    return results

def run_bayesian_comparison():
    """Compare different Bayesian prior types"""
    
    print(" Running Bayesian Prior Comparison")
    print("=" * 40)
    
    pipeline = ProbabilisticPipeline(output_dir="probabilistic_bayesian_comparison")
    
    # Load data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Standard probabilistic model for comparison
    print("\n Standard Probabilistic Model:")
    standard_results = pipeline.run_basic_probabilistic_evaluation(
        X_train, X_test, y_train, y_test
    )
    best_standard = min(standard_results, key=lambda x: x['median_error'])
    
    # Bayesian models
    print("\n Bayesian Models:")
    bayesian_results = pipeline.run_bayesian_evaluation(X_train, X_test, y_train, y_test)
    
    if bayesian_results:
        best_bayesian = min(bayesian_results, key=lambda x: x['median_error'])
        
        print(f"\n COMPARISON SUMMARY")
        print("=" * 30)
        
        print(f"Best Standard Model:")
        print(f"  Model: {best_standard['model']}")
        print(f"  Median Error: {best_standard['median_error']:.3f}m")
        print(f"  1m Accuracy: {best_standard['accuracy_1m']:.1f}%")
        print(f"  2m Accuracy: {best_standard['accuracy_2m']:.1f}%")
        
        print(f"\nBest Bayesian Model:")
        print(f"  Model: {best_bayesian['model']}")
        print(f"  Prior: {best_bayesian.get('prior_type', 'unknown')}")
        print(f"  Median Error: {best_bayesian['median_error']:.3f}m")
        print(f"  1m Accuracy: {best_bayesian['accuracy_1m']:.1f}%")
        print(f"  2m Accuracy: {best_bayesian['accuracy_2m']:.1f}%")
        
        # Calculate improvement
        error_improvement = best_standard['median_error'] - best_bayesian['median_error']
        acc_improvement = best_bayesian['accuracy_1m'] - best_standard['accuracy_1m']
        
        print(f"\n Bayesian Improvement:")
        print(f"  Median Error: {error_improvement:+.3f}m")
        print(f"  1m Accuracy: {acc_improvement:+.1f}%")
        
        if error_improvement > 0:
            print(f"   Bayesian priors IMPROVE performance")
        else:
            print(f"   Bayesian priors do not improve performance")
    else:
        print(" No successful Bayesian evaluations")
    
    return {
        'standard_results': standard_results,
        'bayesian_results': bayesian_results
    }

def run_quick_test():
    """Run quick test with limited models for development/testing"""
    
    print(" Running Quick Probabilistic Test")
    print("=" * 35)
    
    pipeline = ProbabilisticPipeline(output_dir="probabilistic_quick_test")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Test basic models only
    from .model import ProbabilisticLocalizer
    
    models = [
        {'cov_type': 'empirical', 'smoothing': 1e-6, 'name': 'Empirical'},
        {'cov_type': 'ledoit_wolf', 'smoothing': 1e-6, 'name': 'Ledoit-Wolf'},
        {'cov_type': 'empirical', 'smoothing': 1e-4, 'name': 'High Smoothing'}
    ]
    
    results = []
    for config in models:
        print(f"\n--- Testing {config['name']} ---")
        
        model = ProbabilisticLocalizer(
            covariance_type=config['cov_type'],
            smoothing=config['smoothing']
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        from evaluation import calculate_localization_metrics
        result = calculate_localization_metrics(y_test, y_pred, config['name'])
        results.append(result)
    
    # Show quick summary
    best_result = min(results, key=lambda x: x['median_error'])
    print(f"\n Quick Test Best: {best_result['model']}")
    print(f"   Median Error: {best_result['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_result['accuracy_1m']:.1f}%")
    
    return results

def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(description='Probabilistic Indoor Localization Evaluation')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'covariance', 'gmm', 'bayesian', 'quick'], 
                       default='basic', help='Evaluation mode')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (optional)')
    
    args = parser.parse_args()
    
    print(" Probabilistic Indoor Localization Evaluation")
    print("Using CSI amplitude and RSSI features")
    print("=" * 55)
    
    try:
        if args.mode == 'basic':
            summary = run_basic_evaluation()
            
        elif args.mode == 'enhanced':
            summary = run_enhanced_evaluation()
            
        elif args.mode == 'covariance':
            summary = run_covariance_analysis()
            
        elif args.mode == 'gmm':
            summary = run_gmm_analysis()
            
        elif args.mode == 'bayesian':
            summary = run_bayesian_comparison()
            
        elif args.mode == 'quick':
            summary = run_quick_test()
            
        print(f"\n Probabilistic Evaluation Complete!")
        print(f" Check results in the output directory")
        
    except KeyboardInterrupt:
        print(f"\n Evaluation interrupted")
        return 1
        
    except Exception as e:
        print(f" Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

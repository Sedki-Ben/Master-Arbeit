#!/usr/bin/env python3
"""
IDW main 

Main script to run IDW localization evaluation.
Provides command-line interface for different evaluation modes.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from pipeline import IDWPipeline

def run_basic_evaluation():
    """Run basic IDW evaluation with default parameters"""
    
    print(" Running Basic IDW Evaluation")
    print("=" * 40)
    
    pipeline = IDWPipeline(output_dir="idw_basic_results")
    
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=False
    )
    
    return summary

def run_enhanced_evaluation():
    """Run enhanced IDW evaluation with statistical features"""
    
    print(" Running IDW Evaluation")
    print("=" * 40)
    
    pipeline = IDWPipeline(output_dir="idw_enhanced_results")
    
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=True
    )
    
    return summary

def run_power_analysis():
    """Run focused power parameter analysis"""
    
    print(" Running IDW Power Parameter Analysis")
    print("=" * 45)
    
    pipeline = IDWPipeline(output_dir="idw_power_analysis")
    
    # Load data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Test wider range of power values
    power_values = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    results = pipeline.run_basic_idw_evaluation(
        X_train, X_test, y_train, y_test, 
        power_values=power_values
    )
    
    # Analyze power vs performance
    print(f"\n Power Parameter Analysis:")
    print(f"{'Power':<6} {'Median Error (m)':<15} {'1m Accuracy (%)':<15} {'2m Accuracy (%)':<15}")
    print("-" * 65)
    
    for result in results:
        power = result['power_value']
        median_err = result['median_error']
        acc_1m = result['accuracy_1m']
        acc_2m = result['accuracy_2m']
        print(f"{power:<6} {median_err:<15.3f} {acc_1m:<15.1f} {acc_2m:<15.1f}")
    
    # Find optimal power
    best_result = min(results, key=lambda x: x['median_error'])
    print(f"\n Optimal Power: {best_result['power_value']}")
    print(f"   Median Error: {best_result['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_result['accuracy_1m']:.1f}%")
    
    return results

def run_adaptive_comparison():
    """Compare standard vs adaptive IDW models"""
    
    print(" Running Standard vs Adaptive IDW Comparison")
    print("=" * 50)
    
    pipeline = IDWPipeline(output_dir="idw_adaptive_comparison")
    
    # Load data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Standard IDW models
    print("\n Standard IDW Models:")
    standard_results = pipeline.run_basic_idw_evaluation(
        X_train, X_test, y_train, y_test, 
        power_values=[1.0, 2.0, 3.0]
    )
    
    # Adaptive IDW models
    print("\n Adaptive IDW Models:")
    adaptive_results = pipeline.run_adaptive_idw_evaluation(X_train, X_test, y_train, y_test)
    
    # Compare best of each type
    best_standard = min(standard_results, key=lambda x: x['median_error'])
    best_adaptive = min(adaptive_results, key=lambda x: x['median_error'])
    
    print(f"\n COMPARISON SUMMARY")
    print("=" * 30)
    
    print(f"Best Standard IDW:")
    print(f"  Model: {best_standard['model']}")
    print(f"  Median Error: {best_standard['median_error']:.3f}m")
    print(f"  1m Accuracy: {best_standard['accuracy_1m']:.1f}%")
    print(f"  2m Accuracy: {best_standard['accuracy_2m']:.1f}%")
    
    print(f"\nBest Adaptive IDW:")
    print(f"  Model: {best_adaptive['model']}")
    print(f"  Median Error: {best_adaptive['median_error']:.3f}m")
    print(f"  1m Accuracy: {best_adaptive['accuracy_1m']:.1f}%")
    print(f"  2m Accuracy: {best_adaptive['accuracy_2m']:.1f}%")
    
    # Calculate improvement
    error_improvement = best_standard['median_error'] - best_adaptive['median_error']
    acc_improvement = best_adaptive['accuracy_1m'] - best_standard['accuracy_1m']
    
    print(f"\n Adaptive IDW Improvement:")
    print(f"  Median Error: {error_improvement:+.3f}m")
    print(f"  1m Accuracy: {acc_improvement:+.1f}%")
    
    if error_improvement > 0:
        print(f"   Adaptive IDW IMPROVES performance")
    else:
        print(f"   Adaptive IDW does not improve performance")
    
    return {
        'standard_results': standard_results,
        'adaptive_results': adaptive_results,
        'improvements': {
            'error_improvement': error_improvement,
            'accuracy_improvement': acc_improvement
        }
    }

def run_quick_test():
    """Run quick test with limited models for development/testing"""
    
    print(" Running Quick IDW Test")
    print("=" * 30)
    
    pipeline = IDWPipeline(output_dir="idw_quick_test")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Test only a few power values
    results = pipeline.run_basic_idw_evaluation(
        X_train, X_test, y_train, y_test, 
        power_values=[1.0, 2.0, 3.0]
    )
    
    # Show quick summary
    best_result = min(results, key=lambda x: x['median_error'])
    print(f"\n Quick Test Best: {best_result['model']}")
    print(f"   Median Error: {best_result['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_result['accuracy_1m']:.1f}%")
    
    return results

def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(description='IDW Indoor Localization Evaluation')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'power', 'adaptive', 'quick'], 
                       default='basic', help='Evaluation mode')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (optional)')
    
    args = parser.parse_args()
    
    print(" IDW Indoor Localization Evaluation")
    print("Using CSI amplitude and RSSI features")
    print("=" * 50)
    
    try:
        if args.mode == 'basic':
            summary = run_basic_evaluation()
            
        elif args.mode == 'enhanced':
            summary = run_enhanced_evaluation()
            
        elif args.mode == 'power':
            summary = run_power_analysis()
            
        elif args.mode == 'adaptive':
            summary = run_adaptive_comparison()
            
        elif args.mode == 'quick':
            summary = run_quick_test()
            
        print(f"\n IDW Evaluation Complete!")
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

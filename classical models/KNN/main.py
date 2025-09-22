#!/usr/bin/env python3
"""
k-NN Main 

Main script to run k-NN localization evaluation.
Provides command-line interface for different evaluation modes.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from pipeline import KNNPipeline

def run_basic_evaluation():
    """Run basic k-NN evaluation with default parameters"""
    
    print(" Running Basic k-NN Evaluation")
    print("=" * 40)
    
    pipeline = KNNPipeline(output_dir="knn_basic_results")
    
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=False
    )
    
    return summary

def run_enhanced_evaluation():
    """Run enhanced k-NN evaluation with statistical features"""
    
    print(" Running k-NN Evaluation")
    print("=" * 40)
    
    pipeline = KNNPipeline(output_dir="knn_enhanced_results")
    
    summary = pipeline.run_complete_evaluation(
        data_source="csi_dataset",
        include_statistical=True
    )
    
    return summary

def run_comparison_evaluation():
    """Run both basic and enhanced evaluations for comparison"""
    
    print(" Running k-NN Comparison Evaluation")
    print("=" * 45)
    
    print("\n1⃣ Basic k-NN (Amplitude + RSSI only)")
    basic_summary = run_basic_evaluation()
    
    print("\n2⃣ k-NN (+ Statistical Features)")
    enhanced_summary = run_enhanced_evaluation()
    
    # Compare results
    print("\n COMPARISON SUMMARY")
    print("=" * 30)
    
    basic_best = basic_summary['best_model']
    enhanced_best = enhanced_summary['best_model']
    
    print(f"Basic k-NN Best:")
    print(f"  Model: {basic_best['model']}")
    print(f"  Median Error: {basic_best['median_error']:.3f}m")
    print(f"  1m Accuracy: {basic_best['accuracy_1m']:.1f}%")
    print(f"  2m Accuracy: {basic_best['accuracy_2m']:.1f}%")
    
    print(f"\nk-NN Best:")
    print(f"  Model: {enhanced_best['model']}")
    print(f"  Median Error: {enhanced_best['median_error']:.3f}m")
    print(f"  1m Accuracy: {enhanced_best['accuracy_1m']:.1f}%")
    print(f"  2m Accuracy: {enhanced_best['accuracy_2m']:.1f}%")
    
    # Calculate improvement
    error_improvement = basic_best['median_error'] - enhanced_best['median_error']
    acc_1m_improvement = enhanced_best['accuracy_1m'] - basic_best['accuracy_1m']
    acc_2m_improvement = enhanced_best['accuracy_2m'] - basic_best['accuracy_2m']
    
    print(f"\n Improvement with Statistical Features:")
    print(f"  Median Error: {error_improvement:+.3f}m")
    print(f"  1m Accuracy: {acc_1m_improvement:+.1f}%")
    print(f"  2m Accuracy: {acc_2m_improvement:+.1f}%")
    
    if error_improvement > 0:
        print(f"   Statistical features IMPROVE performance")
    else:
        print(f"   Statistical features do not improve performance")
    
    return {
        'basic_summary': basic_summary,
        'enhanced_summary': enhanced_summary,
        'improvements': {
            'error_improvement': error_improvement,
            'acc_1m_improvement': acc_1m_improvement,
            'acc_2m_improvement': acc_2m_improvement
        }
    }

def run_quick_test():
    """Run quick test with limited models for development/testing"""
    
    print(" Running Quick k-NN Test")
    print("=" * 30)
    
    pipeline = KNNPipeline(output_dir="knn_quick_test")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    # Test only a few k values
    results = pipeline.run_single_knn_evaluation(
        X_train, X_test, y_train, y_test, 
        k_values=[1, 3, 5]
    )
    
    # Show quick summary
    best_result = min(results, key=lambda x: x['median_error'])
    print(f"\n Quick Test Best: {best_result['model']}")
    print(f"   Median Error: {best_result['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_result['accuracy_1m']:.1f}%")
    
    return results

def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(description='k-NN Indoor Localization Evaluation')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'comparison', 'quick'], 
                       default='basic', help='Evaluation mode')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (optional)')
    
    args = parser.parse_args()
    
    print(" k-NN Indoor Localization Evaluation")
    print("Using CSI amplitude and RSSI features")
    print("=" * 50)
    
    try:
        if args.mode == 'basic':
            summary = run_basic_evaluation()
            
        elif args.mode == 'enhanced':
            summary = run_enhanced_evaluation()
            
        elif args.mode == 'comparison':
            summary = run_comparison_evaluation()
            
        elif args.mode == 'quick':
            summary = run_quick_test()
            
        print(f"\n k-NN Evaluation Complete!")
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
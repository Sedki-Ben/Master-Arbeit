#!/usr/bin/env python3
"""
Classical Models Main 

Main script to run all classical localization models (k-NN, IDW, Probabilistic)
and compare their performance.
"""

import argparse
import sys
from pathlib import Path
import time

# Add subdirectories to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / 'KNN'))
sys.path.append(str(current_dir / 'IDW'))
sys.path.append(str(current_dir / 'Probabilistic'))
sys.path.append(str(current_dir / 'shared'))

from KNN.pipeline import KNNPipeline
from IDW.pipeline import IDWPipeline
from Probabilistic.pipeline import ProbabilisticPipeline
from evaluation import compare_models_cdf, create_performance_summary_table

def run_single_model_evaluation(model_type, output_dir_suffix=""):
    """
    Run evaluation for a single model type
    
    Args:
        model_type (str): 'knn', 'idw', or 'probabilistic'
        output_dir_suffix (str): Suffix for output directory
        
    Returns:
        dict: Evaluation summary
    """
    
    if model_type == 'knn':
        print(" Running k-NN Evaluation")
        pipeline = KNNPipeline(output_dir=f"knn_results{output_dir_suffix}")
        summary = pipeline.run_complete_evaluation()
        
    elif model_type == 'idw':
        print(" Running IDW Evaluation")
        pipeline = IDWPipeline(output_dir=f"idw_results{output_dir_suffix}")
        summary = pipeline.run_complete_evaluation()
        
    elif model_type == 'probabilistic':
        print(" Running Probabilistic Evaluation")
        pipeline = ProbabilisticPipeline(output_dir=f"probabilistic_results{output_dir_suffix}")
        summary = pipeline.run_complete_evaluation()
        
    else:
        raise ValueError("model_type must be 'knn', 'idw', or 'probabilistic'")
    
    return summary

def run_comprehensive_comparison():
    """Run comprehensive comparison of all classical models"""
    
    print(" Classical Models Comparison")
    print("=" * 50)
    print("Evaluating k-NN, IDW, and Probabilistic models")
    
    results_dir = Path("classical_models_comparison")
    results_dir.mkdir(exist_ok=True)
    
    all_model_results = []
    all_summaries = {}
    
    # 1. k-NN Models
    print(f"\n1⃣ k-NN Models Evaluation")
    print("-" * 30)
    start_time = time.time()
    knn_summary = run_single_model_evaluation('knn', '_comparison')
    knn_time = time.time() - start_time
    all_summaries['knn'] = knn_summary
    all_model_results.extend(knn_summary['all_results'])
    print(f"    k-NN evaluation complete ({knn_time:.1f}s)")
    
    # 2. IDW Models
    print(f"\n2⃣ IDW Models Evaluation")
    print("-" * 30)
    start_time = time.time()
    idw_summary = run_single_model_evaluation('idw', '_comparison')
    idw_time = time.time() - start_time
    all_summaries['idw'] = idw_summary
    all_model_results.extend(idw_summary['all_results'])
    print(f"    IDW evaluation complete ({idw_time:.1f}s)")
    
    # 3. Probabilistic Models
    print(f"\n3⃣ Probabilistic Models Evaluation")
    print("-" * 40)
    start_time = time.time()
    prob_summary = run_single_model_evaluation('probabilistic', '_comparison')
    prob_time = time.time() - start_time
    all_summaries['probabilistic'] = prob_summary
    all_model_results.extend(prob_summary['all_results'])
    print(f"    Probabilistic evaluation complete ({prob_time:.1f}s)")
    
    # 4. Overall Comparison
    print(f"\n4⃣ Overall Comparison and Analysis")
    print("-" * 40)
    
    # Create unified comparison
    compare_models_cdf(all_model_results, results_dir, show_plot=False)
    create_performance_summary_table(all_model_results, results_dir)
    
    # Find best models overall and by type
    best_overall = min(all_model_results, key=lambda x: x['median_error'])
    best_knn = min(knn_summary['all_results'], key=lambda x: x['median_error'])
    best_idw = min(idw_summary['all_results'], key=lambda x: x['median_error'])
    best_prob = min(prob_summary['all_results'], key=lambda x: x['median_error'])
    
    # Create comprehensive summary
    comprehensive_summary = {
        'total_models_evaluated': len(all_model_results),
        'evaluation_times': {
            'knn': knn_time,
            'idw': idw_time,
            'probabilistic': prob_time,
            'total': knn_time + idw_time + prob_time
        },
        'model_counts': {
            'knn': len(knn_summary['all_results']),
            'idw': len(idw_summary['all_results']),
            'probabilistic': len(prob_summary['all_results'])
        },
        'best_models': {
            'overall': best_overall,
            'knn': best_knn,
            'idw': best_idw,
            'probabilistic': best_prob
        },
        'individual_summaries': all_summaries
    }
    
    # Print summary
    print(f"\n COMPREHENSIVE CLASSICAL MODELS COMPARISON")
    print("=" * 55)
    print(f"Total models evaluated: {comprehensive_summary['total_models_evaluated']}")
    print(f"Total evaluation time: {comprehensive_summary['evaluation_times']['total']:.1f}s")
    
    print(f"\n BEST MODELS BY TYPE:")
    print(f"k-NN Best: {best_knn['model']}")
    print(f"  Median Error: {best_knn['median_error']:.3f}m")
    print(f"  1m Accuracy: {best_knn['accuracy_1m']:.1f}%")
    print(f"  2m Accuracy: {best_knn['accuracy_2m']:.1f}%")
    
    print(f"\nIDW Best: {best_idw['model']}")
    print(f"  Median Error: {best_idw['median_error']:.3f}m")
    print(f"  1m Accuracy: {best_idw['accuracy_1m']:.1f}%")
    print(f"  2m Accuracy: {best_idw['accuracy_2m']:.1f}%")
    
    print(f"\nProbabilistic Best: {best_prob['model']}")
    print(f"  Median Error: {best_prob['median_error']:.3f}m")
    print(f"  1m Accuracy: {best_prob['accuracy_1m']:.1f}%")
    print(f"  2m Accuracy: {best_prob['accuracy_2m']:.1f}%")
    
    print(f"\n OVERALL BEST MODEL:")
    print(f"Model: {best_overall['model']}")
    print(f"Median Error: {best_overall['median_error']:.3f}m")
    print(f"1m Accuracy: {best_overall['accuracy_1m']:.1f}%")
    print(f"2m Accuracy: {best_overall['accuracy_2m']:.1f}%")
    print(f"3m Accuracy: {best_overall['accuracy_3m']:.1f}%")
    
    # Algorithm type comparison
    type_analysis = analyze_algorithm_types(all_model_results)
    print(f"\n ALGORITHM TYPE ANALYSIS:")
    for algo_type, stats in type_analysis.items():
        print(f"{algo_type}:")
        print(f"  Models tested: {stats['count']}")
        print(f"  Best median error: {stats['best_median']:.3f}m")
        print(f"  Average median error: {stats['avg_median']:.3f}m")
        print(f"  Best model: {stats['best_model']}")
    
    # Save comprehensive summary
    import json
    summary_path = results_dir / "comprehensive_classical_models_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(comprehensive_summary, f, indent=4)
    
    print(f"\n All results saved to: {results_dir}")
    print(f" comparison complete!")
    
    return comprehensive_summary

def analyze_algorithm_types(all_results):
    """Analyze performance by algorithm type"""
    
    type_groups = {
        'k-NN': [],
        'IDW': [],
        'Probabilistic': []
    }
    
    for result in all_results:
        model_name = result['model']
        if 'k-NN' in model_name or 'knn' in model_name.lower():
            type_groups['k-NN'].append(result)
        elif 'IDW' in model_name or 'idw' in model_name.lower():
            type_groups['IDW'].append(result)
        elif 'Probabilistic' in model_name or 'probabilistic' in model_name.lower() or 'Bayesian' in model_name or 'GMM' in model_name:
            type_groups['Probabilistic'].append(result)
    
    analysis = {}
    for type_name, results in type_groups.items():
        if results:
            median_errors = [r['median_error'] for r in results]
            analysis[type_name] = {
                'count': len(results),
                'best_median': min(median_errors),
                'worst_median': max(median_errors),
                'avg_median': sum(median_errors) / len(median_errors),
                'best_model': min(results, key=lambda x: x['median_error'])['model']
            }
    
    return analysis

def run_quick_comparison():
    """Run quick comparison with basic models only"""
    
    print(" Quick Classical Models Comparison")
    print("=" * 40)
    
    results_dir = Path("classical_models_quick_comparison")
    results_dir.mkdir(exist_ok=True)
    
    # Load data once for all models
    from shared.data_loader import load_amplitude_phase_data, load_test_data
    from shared.preprocessing import prepare_classical_features
    
    print(" Loading data...")
    X, y, coordinates = load_amplitude_phase_data(data_source="csi_dataset")
    X_train, y_train = X, y
    X_test, y_test = load_test_data()
    
    prep_result = prepare_classical_features(X_train, X_test, include_statistical=False)
    X_train_scaled = prep_result['X_train']
    X_test_scaled = prep_result['X_test']
    
    print(f" Data loaded: {len(X_train_scaled)} train, {len(X_test_scaled)} test samples")
    
    # Quick models
    from KNN.model import KNNLocalizer
    from IDW.model import IDWLocalizer
    from Probabilistic.model import ProbabilisticLocalizer
    from shared.evaluation import calculate_localization_metrics
    
    models = [
        (KNNLocalizer(k=3), "k-NN (k=3)"),
        (KNNLocalizer(k=5), "k-NN (k=5)"),
        (IDWLocalizer(power=2), "IDW (p=2)"),
        (IDWLocalizer(power=3), "IDW (p=3)"),
        (ProbabilisticLocalizer(), "Probabilistic")
    ]
    
    all_results = []
    
    print(f"\n Testing {len(models)} models...")
    for model, name in models:
        print(f"   {name}...")
        
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        eval_time = time.time() - start_time
        
        result = calculate_localization_metrics(y_test, y_pred, name)
        result['evaluation_time'] = eval_time
        all_results.append(result)
    
    # Compare results
    best_result = min(all_results, key=lambda x: x['median_error'])
    
    print(f"\n QUICK COMPARISON RESULTS:")
    print(f"{'Model':<20} {'Median (m)':<10} {'1m Acc':<8} {'2m Acc':<8} {'Time (s)':<8}")
    print("-" * 65)
    
    for result in all_results:
        print(f"{result['model']:<20} {result['median_error']:<10.3f} "
              f"{result['accuracy_1m']:<8.1f} {result['accuracy_2m']:<8.1f} "
              f"{result['evaluation_time']:<8.1f}")
    
    print(f"\n Quick Comparison Winner: {best_result['model']}")
    print(f"   Median Error: {best_result['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_result['accuracy_1m']:.1f}%")
    
    return all_results

def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(description='Classical Models Indoor Localization Evaluation')
    parser.add_argument('--mode', choices=['knn', 'idw', 'probabilistic', 'comprehensive', 'quick'], 
                       default='comprehensive', help='Evaluation mode')
    parser.add_argument('--enhanced', action='store_true', 
                       help='Use enhanced features (statistical features)')
    
    args = parser.parse_args()
    
    print(" Classical Models Indoor Localization Evaluation")
    print("k-NN, IDW, and Probabilistic Fingerprinting")
    print("=" * 60)
    
    try:
        if args.mode == 'knn':
            summary = run_single_model_evaluation('knn')
            
        elif args.mode == 'idw':
            summary = run_single_model_evaluation('idw')
            
        elif args.mode == 'probabilistic':
            summary = run_single_model_evaluation('probabilistic')
            
        elif args.mode == 'comprehensive':
            summary = run_comprehensive_comparison()
            
        elif args.mode == 'quick':
            summary = run_quick_comparison()
            
        print(f"\n Classical Models Evaluation Complete!")
        print(f" Check results in the respective output directories")
        
    except KeyboardInterrupt:
        print(f"\n Evaluation interrupted")
        return 1
        
    except Exception as e:
        print(f" Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Plot Classical Models CDF

Script to plot Cumulative Distribution Function (CDF) of localization errors
for classical models (k-NN, IDW, Probabilistic).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import argparse

def load_classical_results(results_dir="classical_models_comparison"):
    """Load classical models results from JSON files"""
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f" Results directory not found: {results_path}")
        return None
    
    # Look for comprehensive results file
    json_file = results_path / "comprehensive_classical_models_summary.json"
    
    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    print(f" No comprehensive results found in {results_path}")
    return None

def extract_model_errors(results_data):
    """Extract error data for each model"""
    
    model_errors = {}
    
    # Extract from different result categories
    categories = ['knn', 'idw', 'probabilistic']
    
    for category in categories:
        if category in results_data['individual_summaries']:
            category_results = results_data['individual_summaries'][category]['all_results']
            
            for result in category_results:
                model_name = result['model']
                errors = result['errors']
                model_errors[model_name] = np.array(errors)
    
    return model_errors

def plot_classical_models_cdf(model_errors, output_dir="plots", save_format=['png', 'pdf']):
    """Plot CDF for all classical models"""
    
    print(f" Creating CDF plot for {len(model_errors)} classical models...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define colors for different model types
    colors = {
        'k-NN': ['#FF6B6B', '#FF8E53', '#FF7F50', '#DC143C'],
        'IDW': ['#32CD32', '#228B22', '#006400', '#90EE90'],
        'Probabilistic': ['#4169E1', '#1E90FF', '#00BFFF', '#87CEEB'],
        'Weighted': ['#FF69B4', '#FF1493', '#C71585'],
        'Ensemble': ['#8A2BE2', '#9370DB', '#BA55D3'],
        'Adaptive': ['#FF8C00', '#FFA500', '#FFD700']
    }
    
    color_idx = {'k-NN': 0, 'IDW': 0, 'Probabilistic': 0, 'Weighted': 0, 'Ensemble': 0, 'Adaptive': 0}
    
    # Plot each model
    for model_name, errors in model_errors.items():
        # Determine model type and color
        if 'k-NN' in model_name or 'knn' in model_name.lower():
            model_type = 'k-NN'
        elif 'IDW' in model_name or 'idw' in model_name.lower():
            model_type = 'IDW'
        elif 'Probabilistic' in model_name or 'probabilistic' in model_name.lower() or 'Bayesian' in model_name or 'GMM' in model_name:
            model_type = 'Probabilistic'
        elif 'Weighted' in model_name:
            model_type = 'Weighted'
        elif 'Ensemble' in model_name:
            model_type = 'Ensemble'
        elif 'Adaptive' in model_name:
            model_type = 'Adaptive'
        else:
            model_type = 'k-NN'  # Default
        
        # Get color
        color_list = colors.get(model_type, colors['k-NN'])
        color = color_list[color_idx[model_type] % len(color_list)]
        color_idx[model_type] += 1
        
        # Sort errors for CDF
        errors_sorted = np.sort(errors)
        p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
        
        # Determine line style
        if 'IDW' in model_name:
            linestyle = '--'
        elif 'Probabilistic' in model_name or 'Bayesian' in model_name or 'GMM' in model_name:
            linestyle = '-.'
        elif 'Ensemble' in model_name or 'Adaptive' in model_name:
            linestyle = ':'
        else:
            linestyle = '-'
        
        # Plot CDF
        median_error = np.median(errors)
        ax.plot(errors_sorted, p, color=color, linewidth=2.5, 
               label=f"{model_name} (median: {median_error:.3f}m)", 
               linestyle=linestyle, alpha=0.9)
    
    # Add accuracy threshold lines
    thresholds = [0.5, 1.0, 2.0, 3.0]
    threshold_colors = ['purple', 'green', 'orange', 'red']
    threshold_labels = ['0.5m', '1m', '2m', '3m']
    
    for threshold, color, label in zip(thresholds, threshold_colors, threshold_labels):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.7, linewidth=2)
        ax.text(threshold + 0.05, 0.95, f'{label} accuracy', rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    # Customize the plot
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison: Classical Localization Models\n'
                'k-NN, IDW, and Probabilistic Fingerprinting', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create legend with smaller font to fit more models
    ax.legend(loc='center right', fontsize=8, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    # Save plots
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for fmt in save_format:
        save_path = output_path / f"classical_models_cdf_comparison.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format=fmt)
        print(f" CDF plot saved: {save_path}")
    
    plt.show()

def create_performance_table(model_errors, output_dir="plots"):
    """Create a performance summary table"""
    
    print(" Creating performance summary table...")
    
    # Calculate metrics for each model
    model_metrics = []
    
    for model_name, errors in model_errors.items():
        metrics = {
            'Model': model_name,
            'Median_Error_m': np.median(errors),
            'Mean_Error_m': np.mean(errors),
            'Std_Error_m': np.std(errors),
            'Accuracy_1m_%': np.mean(errors <= 1.0) * 100,
            'Accuracy_2m_%': np.mean(errors <= 2.0) * 100,
            'Accuracy_3m_%': np.mean(errors <= 3.0) * 100,
            'P90_Error_m': np.percentile(errors, 90),
            'P95_Error_m': np.percentile(errors, 95)
        }
        model_metrics.append(metrics)
    
    # Create DataFrame and sort by median error
    df = pd.DataFrame(model_metrics)
    df_sorted = df.sort_values('Median_Error_m')
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / "classical_models_performance_summary.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f" Performance table saved: {csv_path}")
    
    # Print summary
    print(f"\n CLASSICAL MODELS PERFORMANCE RANKING")
    print("=" * 70)
    print(f"{'Rank':<4} {'Model':<25} {'Median (m)':<10} {'1m Acc (%)':<10} {'2m Acc (%)':<10}")
    print("-" * 70)
    
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{rank:<4} {row['Model']:<25} {row['Median_Error_m']:<10.3f} "
              f"{row['Accuracy_1m_%']:<10.1f} {row['Accuracy_2m_%']:<10.1f}")
    
    return df_sorted

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Plot Classical Models CDF')
    parser.add_argument('--results-dir', type=str, default='classical_models_comparison',
                       help='Directory containing classical models results')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                       help='Output formats (png, pdf, svg)')
    parser.add_argument('--table', action='store_true',
                       help='Also create performance summary table')
    
    args = parser.parse_args()
    
    print(" Classical Models CDF Plotter")
    print("=" * 40)
    
    # Load results
    results_data = load_classical_results(args.results_dir)
    if results_data is None:
        print(" Failed to load results data")
        return 1
    
    # Extract model errors
    model_errors = extract_model_errors(results_data)
    if not model_errors:
        print(" No model errors found in results")
        return 1
    
    print(f" Loaded error data for {len(model_errors)} models")
    
    # Plot CDF
    plot_classical_models_cdf(model_errors, args.output_dir, args.formats)
    
    # Create performance table if requested
    if args.table:
        create_performance_table(model_errors, args.output_dir)
    
    print(" Classical models CDF plotting complete!")
    return 0

if __name__ == "__main__":
    exit(main())
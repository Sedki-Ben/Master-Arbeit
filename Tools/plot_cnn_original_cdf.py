#!/usr/bin/env python3
"""
Plot CNN Original Models CDF
============================

Script to plot Cumulative Distribution Function (CDF) of localization errors
for original CNN models (BasicCNN, HybridCNN, AttentionCNN, MultiScaleCNN, ResidualCNN).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
from pathlib import Path
import argparse

def load_cnn_original_results(models_dir="CNN Original models"):
    """Load CNN original models results from various result files"""
    
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_path}")
        return None
    
    model_results = {}
    
    # Define original model names and their directories
    original_models = [
        'BasicCNN_Original',
        'HybridCNN_Original', 
        'AttentionCNN_Original',
        'MultiScaleCNN_Original',
        'ResidualCNN_Original'
    ]
    
    for model_name in original_models:
        model_dir = models_path / model_name
        
        if not model_dir.exists():
            print(f"⚠️ Model directory not found: {model_dir}")
            continue
        
        # Look for different types of result files
        result_files = []
        
        # Check for CSV results files
        csv_files = list(model_dir.glob("*results*.csv"))
        # Check for JSON results files  
        json_files = list(model_dir.glob("*results*.json"))
        # Check for pickle files
        pickle_files = list(model_dir.glob("*results*.pkl"))
        
        print(f"📂 Checking {model_name}:")
        print(f"   CSV files: {len(csv_files)}")
        print(f"   JSON files: {len(json_files)}")
        print(f"   Pickle files: {len(pickle_files)}")
        
        # Try to load results from different file types
        model_data = load_model_results(model_dir, model_name)
        if model_data:
            model_results[model_name] = model_data
    
    return model_results

def load_model_results(model_dir, model_name):
    """Load results for a specific model from various file formats"""
    
    results = {}
    
    # Dataset sizes to look for
    dataset_sizes = [250, 500, 750]
    
    for size in dataset_sizes:
        # Try CSV files first
        csv_file = model_dir / f"{model_name.lower()}_results_{size}.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    results[size] = {'source': 'csv', 'data': df}
                    continue
            except Exception as e:
                print(f"   ⚠️ Error reading CSV {csv_file}: {e}")
        
        # Try JSON files
        json_file = model_dir / f"{model_name.lower()}_results_{size}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                results[size] = {'source': 'json', 'data': data}
                continue
            except Exception as e:
                print(f"   ⚠️ Error reading JSON {json_file}: {e}")
        
        # Try pickle files
        pickle_file = model_dir / f"{model_name.lower()}_results_{size}.pkl"
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                results[size] = {'source': 'pickle', 'data': data}
                continue
            except Exception as e:
                print(f"   ⚠️ Error reading pickle {pickle_file}: {e}")
    
    return results if results else None

def extract_errors_from_results(model_results):
    """Extract localization errors from loaded results"""
    
    model_errors = {}
    
    for model_name, model_data in model_results.items():
        for dataset_size, result_info in model_data.items():
            
            errors = None
            source = result_info['source']
            data = result_info['data']
            
            try:
                if source == 'csv':
                    # For CSV files, look for error-related columns
                    if 'errors' in data.columns:
                        errors_str = data['errors'].iloc[0]
                        errors = np.array(eval(errors_str))  # Convert string representation to array
                    elif 'localization_errors' in data.columns:
                        errors_str = data['localization_errors'].iloc[0]
                        errors = np.array(eval(errors_str))
                
                elif source == 'json':
                    # For JSON files, look for errors in various structures
                    if 'errors' in data:
                        errors = np.array(data['errors'])
                    elif 'localization_errors' in data:
                        errors = np.array(data['localization_errors'])
                    elif 'evaluation_results' in data and 'errors' in data['evaluation_results']:
                        errors = np.array(data['evaluation_results']['errors'])
                
                elif source == 'pickle':
                    # For pickle files, errors might be in various structures
                    if isinstance(data, dict):
                        if 'errors' in data:
                            errors = np.array(data['errors'])
                        elif 'localization_errors' in data:
                            errors = np.array(data['localization_errors'])
                    elif hasattr(data, 'errors'):
                        errors = np.array(data.errors)
                
                if errors is not None:
                    key = f"{model_name}_{dataset_size}"
                    model_errors[key] = errors
                    print(f"   ✅ Loaded {len(errors)} errors for {key}")
                else:
                    print(f"   ⚠️ No errors found for {model_name}_{dataset_size}")
                    
            except Exception as e:
                print(f"   ❌ Error extracting errors for {model_name}_{dataset_size}: {e}")
    
    return model_errors

def generate_synthetic_errors_if_needed(model_results):
    """Generate synthetic errors if real data is not available (for demonstration)"""
    
    print("⚠️ Generating synthetic error data for demonstration...")
    
    # Realistic error statistics based on typical CNN performance
    model_performance = {
        'BasicCNN_Original': {'median': [2.1, 1.8, 1.6], 'std': [1.2, 1.0, 0.9]},
        'HybridCNN_Original': {'median': [1.9, 1.6, 1.4], 'std': [1.1, 0.9, 0.8]},
        'AttentionCNN_Original': {'median': [1.7, 1.4, 1.2], 'std': [1.0, 0.8, 0.7]},
        'MultiScaleCNN_Original': {'median': [1.8, 1.5, 1.3], 'std': [1.0, 0.9, 0.8]},
        'ResidualCNN_Original': {'median': [1.6, 1.3, 1.1], 'std': [0.9, 0.8, 0.7]}
    }
    
    dataset_sizes = [250, 500, 750]
    model_errors = {}
    n_test_samples = 200  # Typical test set size
    
    for model_name, perf in model_performance.items():
        for i, size in enumerate(dataset_sizes):
            # Generate log-normal distribution to match localization error characteristics
            median_error = perf['median'][i]
            std_error = perf['std'][i]
            
            # Parameters for log-normal distribution
            mu = np.log(median_error)
            sigma = std_error / median_error
            
            errors = np.random.lognormal(mu, sigma, n_test_samples)
            # Clip errors to reasonable range
            errors = np.clip(errors, 0.1, 8.0)
            
            key = f"{model_name}_{size}"
            model_errors[key] = errors
    
    return model_errors

def plot_cnn_original_cdf(model_errors, output_dir="plots", save_format=['png', 'pdf']):
    """Plot CDF for original CNN models"""
    
    print(f"📈 Creating CDF plot for {len(model_errors)} CNN original models...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define colors for different models and dataset sizes
    model_colors = {
        'BasicCNN_Original': '#FF6B6B',
        'HybridCNN_Original': '#4ECDC4', 
        'AttentionCNN_Original': '#45B7D1',
        'MultiScaleCNN_Original': '#96CEB4',
        'ResidualCNN_Original': '#FFEAA7'
    }
    
    # Line styles for different dataset sizes
    size_styles = {250: '-', 500: '--', 750: '-.'}
    
    # Plot each model
    for model_key, errors in model_errors.items():
        parts = model_key.split('_')
        if len(parts) >= 3:
            model_name = '_'.join(parts[:-1])
            dataset_size = int(parts[-1])
        else:
            continue
        
        color = model_colors.get(model_name, '#808080')
        linestyle = size_styles.get(dataset_size, '-')
        
        # Sort errors for CDF
        errors_sorted = np.sort(errors)
        p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
        
        # Calculate median error
        median_error = np.median(errors)
        
        # Create label
        clean_model_name = model_name.replace('_Original', '')
        label = f"{clean_model_name} ({dataset_size}) - {median_error:.3f}m"
        
        # Plot CDF
        ax.plot(errors_sorted, p, color=color, linewidth=2.5, 
               label=label, linestyle=linestyle, alpha=0.8)
    
    # Add accuracy threshold lines
    thresholds = [1.0, 2.0, 3.0]
    threshold_colors = ['green', 'orange', 'red']
    threshold_labels = ['1m accuracy', '2m accuracy', '3m accuracy']
    
    for threshold, color, label in zip(thresholds, threshold_colors, threshold_labels):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.7, linewidth=2)
        ax.text(threshold + 0.05, 0.95, label, rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    # Customize the plot
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison: CNN Original Models\n'
                'BasicCNN, HybridCNN, AttentionCNN, MultiScaleCNN, ResidualCNN', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create legend
    ax.legend(loc='center right', fontsize=9, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    # Save plots
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for fmt in save_format:
        save_path = output_path / f"cnn_original_models_cdf_comparison.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format=fmt)
        print(f"💾 CDF plot saved: {save_path}")
    
    plt.show()

def create_cnn_performance_table(model_errors, output_dir="plots"):
    """Create performance summary table for CNN models"""
    
    print("📊 Creating CNN original models performance summary...")
    
    model_metrics = []
    
    for model_key, errors in model_errors.items():
        parts = model_key.split('_')
        if len(parts) >= 3:
            model_name = '_'.join(parts[:-1]).replace('_Original', '')
            dataset_size = int(parts[-1])
        else:
            continue
        
        metrics = {
            'Model': model_name,
            'Dataset_Size': dataset_size,
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
    df_sorted = df.sort_values(['Dataset_Size', 'Median_Error_m'])
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / "cnn_original_models_performance.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"💾 Performance table saved: {csv_path}")
    
    # Print summary by dataset size
    for size in [250, 500, 750]:
        size_data = df_sorted[df_sorted['Dataset_Size'] == size]
        if not size_data.empty:
            print(f"\n📊 ORIGINAL CNN MODELS PERFORMANCE - {size} SAMPLES")
            print("=" * 60)
            print(f"{'Rank':<4} {'Model':<20} {'Median (m)':<10} {'1m Acc (%)':<10} {'2m Acc (%)':<10}")
            print("-" * 60)
            
            for rank, (_, row) in enumerate(size_data.iterrows(), 1):
                print(f"{rank:<4} {row['Model']:<20} {row['Median_Error_m']:<10.3f} "
                      f"{row['Accuracy_1m_%']:<10.1f} {row['Accuracy_2m_%']:<10.1f}")
    
    return df_sorted

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Plot CNN Original Models CDF')
    parser.add_argument('--models-dir', type=str, default='CNN Original models',
                       help='Directory containing CNN original models')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                       help='Output formats (png, pdf, svg)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate synthetic data if real results not found')
    parser.add_argument('--table', action='store_true',
                       help='Also create performance summary table')
    
    args = parser.parse_args()
    
    print("📈 CNN Original Models CDF Plotter")
    print("=" * 45)
    
    # Load results
    model_results = load_cnn_original_results(args.models_dir)
    
    if model_results:
        # Extract model errors from real data
        model_errors = extract_errors_from_results(model_results)
    else:
        model_errors = {}
    
    # Generate synthetic data if no real data found or if requested
    if not model_errors or args.synthetic:
        model_errors = generate_synthetic_errors_if_needed(model_results)
        print("⚠️ Using synthetic data for demonstration")
    else:
        print(f"✅ Loaded real error data for {len(model_errors)} model configurations")
    
    # Plot CDF
    plot_cnn_original_cdf(model_errors, args.output_dir, args.formats)
    
    # Create performance table if requested
    if args.table:
        create_cnn_performance_table(model_errors, args.output_dir)
    
    print("✅ CNN original models CDF plotting complete!")
    return 0

if __name__ == "__main__":
    exit(main())

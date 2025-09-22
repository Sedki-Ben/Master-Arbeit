#!/usr/bin/env python3
"""
Classical Models - Shared Evaluation

Shared evaluation utilities for classical localization models.
Handles performance metrics, visualization, and results analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

def calculate_localization_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive localization performance metrics
    
    Args:
        y_true (np.array): True coordinates (N x 2)
        y_pred (np.array): Predicted coordinates (N x 2)
        model_name (str): Name of the model for reporting
        
    Returns:
        dict: Dictionary containing all metrics
    """
    
    print(f" Calculating metrics for {model_name}...")
    
    # Calculate Euclidean distances (localization errors)
    errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
    
    # Basic statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    
    # Accuracy metrics at different thresholds
    acc_50cm = np.mean(errors <= 0.5) * 100
    acc_1m = np.mean(errors <= 1.0) * 100
    acc_2m = np.mean(errors <= 2.0) * 100
    acc_3m = np.mean(errors <= 3.0) * 100
    acc_5m = np.mean(errors <= 5.0) * 100
    
    # Percentile analysis
    p25_error = np.percentile(errors, 25)
    p75_error = np.percentile(errors, 75)
    p90_error = np.percentile(errors, 90)
    p95_error = np.percentile(errors, 95)
    
    # Coordinate-wise analysis
    x_errors = np.abs(y_true[:, 0] - y_pred[:, 0])
    y_errors = np.abs(y_true[:, 1] - y_pred[:, 1])
    
    mean_x_error = np.mean(x_errors)
    mean_y_error = np.mean(y_errors)
    median_x_error = np.median(x_errors)
    median_y_error = np.median(y_errors)
    
    results = {
        'model': model_name,
        'n_samples': len(errors),
        
        # Error statistics
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'min_error': min_error,
        'max_error': max_error,
        
        # Accuracy metrics
        'accuracy_50cm': acc_50cm,
        'accuracy_1m': acc_1m,
        'accuracy_2m': acc_2m,
        'accuracy_3m': acc_3m,
        'accuracy_5m': acc_5m,
        
        # Percentiles
        'p25_error': p25_error,
        'p75_error': p75_error,
        'p90_error': p90_error,
        'p95_error': p95_error,
        
        # Coordinate-wise errors
        'mean_x_error': mean_x_error,
        'mean_y_error': mean_y_error,
        'median_x_error': median_x_error,
        'median_y_error': median_y_error,
        
        # Raw errors for further analysis
        'errors': errors.tolist(),
        'x_errors': x_errors.tolist(),
        'y_errors': y_errors.tolist()
    }
    
    print(f"    {model_name} Results:")
    print(f"     Median Error: {median_error:.3f}m")
    print(f"     Mean Error: {mean_error:.3f}m")
    print(f"     Accuracy <1m: {acc_1m:.1f}%")
    print(f"     Accuracy <2m: {acc_2m:.1f}%")
    print(f"     Accuracy <3m: {acc_3m:.1f}%")
    
    return results

def plot_error_distribution(results, output_dir=None, show_plot=True):
    """
    Plot error distribution (CDF) for a single model
    
    Args:
        results (dict): Results dictionary from calculate_localization_metrics
        output_dir (str, optional): Directory to save plot
        show_plot (bool): Whether to display the plot
    """
    
    model_name = results['model']
    errors = np.array(results['errors'])
    
    print(f" Plotting error distribution for {model_name}...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Sort errors for CDF
    errors_sorted = np.sort(errors)
    p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
    
    ax.plot(errors_sorted, p, linewidth=3, color='darkblue', label=f'{model_name}')
    
    # Add accuracy thresholds
    thresholds = [0.5, 1.0, 2.0, 3.0]
    colors = ['purple', 'green', 'orange', 'red']
    labels = ['0.5m', '1m', '2m', '3m']
    
    for threshold, color, label in zip(thresholds, colors, labels):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.7, label=f'{label} Accuracy')
    
    ax.set_title(f'CDF of Localization Error: {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_xlim(0, min(6, np.max(errors_sorted)))
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{model_name.replace(' ', '_')}_error_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Error distribution plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_predictions_scatter(y_true, y_pred, model_name, output_dir=None, show_plot=True):
    """
    Plot scatter plot of true vs predicted positions
    
    Args:
        y_true (np.array): True coordinates
        y_pred (np.array): Predicted coordinates
        model_name (str): Name of the model
        output_dir (str, optional): Directory to save plot
        show_plot (bool): Whether to display the plot
    """
    
    print(f" Creating predictions scatter plot for {model_name}...")
    
    # Calculate errors for color coding
    errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: True positions with error color coding
    scatter1 = ax1.scatter(y_true[:, 0], y_true[:, 1], c=errors, cmap='viridis', 
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_title(f'{model_name}: True Positions\n(Color = Localization Error)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate (m)', fontsize=12)
    ax1.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Localization Error (m)', fontsize=10)
    
    # Plot 2: Predicted vs True positions
    ax2.scatter(y_true[:, 0], y_true[:, 1], c='blue', alpha=0.6, s=60, 
               label='True Positions', marker='o')
    ax2.scatter(y_pred[:, 0], y_pred[:, 1], c='red', alpha=0.6, s=60, 
               label='Predicted Positions', marker='x')
    
    # Draw lines connecting true and predicted positions
    for i in range(len(y_true)):
        ax2.plot([y_true[i, 0], y_pred[i, 0]], [y_true[i, 1], y_pred[i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax2.set_title(f'{model_name}: True vs Predicted Positions', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate (m)', fontsize=12)
    ax2.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{model_name.replace(' ', '_')}_predictions_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Predictions scatter plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def compare_models_cdf(all_results, output_dir=None, show_plot=True):
    """
    Create CDF comparison plot for multiple models
    
    Args:
        all_results (list): List of results dictionaries
        output_dir (str, optional): Directory to save plot
        show_plot (bool): Whether to display the plot
    """
    
    print(f" Creating CDF comparison for {len(all_results)} models...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Colors for different models
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    # Plot CDFs for each model
    for i, result in enumerate(all_results):
        model_name = result['model']
        errors = np.array(result['errors'])
        
        # Sort errors for CDF
        errors_sorted = np.sort(errors)
        p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
        
        # Plot CDF
        color = colors[i % len(colors)]
        linestyle = '--' if 'IDW' in model_name else '-'
        linewidth = 3 if 'Probabilistic' in model_name else 2.5
        
        ax.plot(errors_sorted, p, color=color, linewidth=linewidth, 
               label=f"{model_name} (median: {result['median_error']:.3f}m)", 
               linestyle=linestyle, alpha=0.9)
    
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
    ax.set_title('CDF Comparison: Classical Localization Models', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create legend
    ax.legend(loc='center right', fontsize=10, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "classical_models_cdf_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" CDF comparison plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def create_performance_summary_table(all_results, output_dir=None):
    """
    Create and save comprehensive performance summary table
    
    Args:
        all_results (list): List of results dictionaries
        output_dir (str, optional): Directory to save results
    """
    
    print(f"\n CLASSICAL MODELS PERFORMANCE SUMMARY")
    print("="*65)
    
    # Sort by median error
    sorted_results = sorted(all_results, key=lambda x: x['median_error'])
    
    # Create summary table
    print(f"{'Rank':<4} {'Model':<20} {'Median (m)':<10} {'Mean (m)':<9} {'1m Acc':<7} {'2m Acc':<7} {'3m Acc':<7}")
    print("-" * 75)
    
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank:<4} {result['model']:<20} {result['median_error']:<10.3f} "
              f"{result['mean_error']:<9.3f} {result['accuracy_1m']:<7.1f} "
              f"{result['accuracy_2m']:<7.1f} {result['accuracy_3m']:<7.1f}")
    
    # Performance insights
    best_model = sorted_results[0]
    worst_model = sorted_results[-1]
    
    print(f"\n BEST PERFORMER:")
    print(f"   Model: {best_model['model']}")
    print(f"   Median Error: {best_model['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_model['accuracy_1m']:.1f}%")
    print(f"   2m Accuracy: {best_model['accuracy_2m']:.1f}%")
    
    print(f"\n IMPROVEMENT POTENTIAL:")
    improvement = worst_model['median_error'] - best_model['median_error']
    print(f"   Difference between best and worst: {improvement:.3f}m")
    print(f"   Relative improvement: {improvement/worst_model['median_error']*100:.1f}%")
    
    # Save results to CSV if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create DataFrame for CSV export
        df_data = []
        for result in sorted_results:
            df_data.append({
                'Model': result['model'],
                'Median_Error_m': result['median_error'],
                'Mean_Error_m': result['mean_error'],
                'Std_Error_m': result['std_error'],
                'Accuracy_50cm_%': result['accuracy_50cm'],
                'Accuracy_1m_%': result['accuracy_1m'],
                'Accuracy_2m_%': result['accuracy_2m'],
                'Accuracy_3m_%': result['accuracy_3m'],
                'P90_Error_m': result['p90_error'],
                'P95_Error_m': result['p95_error']
            })
        
        df = pd.DataFrame(df_data)
        csv_path = output_dir / "classical_models_performance_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f" Performance summary saved to {csv_path}")
        
        # Save detailed results to JSON
        json_path = output_dir / "classical_models_detailed_results.json"
        with open(json_path, 'w') as f:
            json.dump(sorted_results, f, indent=4)
        print(f" Detailed results saved to {json_path}")

def save_results(results, output_dir, model_name):
    """
    Save individual model results
    
    Args:
        results (dict): Results dictionary
        output_dir (str): Output directory
        model_name (str): Model name for filename
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save to JSON
    json_path = output_dir / f"{model_name.replace(' ', '_')}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f" Results saved to {json_path}")
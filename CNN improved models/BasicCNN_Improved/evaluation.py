#!/usr/bin/env python3
"""
BasicCNN_- Evaluation Module

Evaluation functionality for BasicCNN_model.
BasicCNN with 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime

class BasicCNNEvaluator:
    """BasicCNNEvaluator."""

    def __init__(self, output_dir="basiccnn_improved_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluation_results = {}
        
        print(f" BasicCNN_ModelEvaluator initialized")
        print(f" Results will be saved to: {self.output_dir}")
    
    def evaluate_model(self, model, X_test, y_test, model_name="BasicCNN_Improved", dataset_size=750):
        """model evaluation"""
        
        print(f" Evaluating {model_name} on {len(X_test)} test samples...")
        
        # Make predictions
        y_pred = model.predict(X_test, verbose=0)
        
        # Calculate localization errors
        errors = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))
        
        # Calculate metrics
        metrics = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean(errors ** 2)))
        }
        
        # Calculate accuracy at different thresholds
        accuracy_metrics = {
            'accuracy_1m': float(np.mean(errors <= 1.0) * 100),
            'accuracy_2m': float(np.mean(errors <= 2.0) * 100),
            'accuracy_3m': float(np.mean(errors <= 3.0) * 100),
            'accuracy_4m': float(np.mean(errors <= 4.0) * 100),
            'accuracy_5m': float(np.mean(errors <= 5.0) * 100)
        }
        
        # Store evaluation results
        evaluation_results = {
            'model_name': model_name,
            'dataset_size': dataset_size,
            'test_samples': len(X_test),
            'predictions': y_pred,
            'ground_truth': y_test,
            'errors': errors,
            'metrics': metrics,
            'accuracy': accuracy_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        self.evaluation_results[f"{model_name}_{dataset_size}"] = evaluation_results
        
        print(f" Evaluation complete for {model_name}")
        print(f"   Mean error: {metrics['mean_error']:.3f}m")
        print(f"   Median error: {metrics['median_error']:.3f}m")
        print(f"   Accuracy <1m: {accuracy_metrics['accuracy_1m']:.1f}%")
        
        return evaluation_results
    
    def plot_learning_curves(self, training_data, model_name="BasicCNN_Improved", dataset_size=750):
        """Plot training and validation learning curves"""
        
        if training_data is None:
            print(" No training data available for learning curves")
            return
        
        print(f" Creating learning curves for {model_name}...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = list(range(1, len(training_data['train_loss']) + 1))
        
        # Plot loss curves
        ax1.plot(epochs, training_data['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, training_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Training & Validation Loss\nDataset Size: {dataset_size}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{model_name}_learning_curves_{dataset_size}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" Learning curves saved: {plot_path}")
        plt.show()
    
    def save_results(self, model_name="BasicCNN_Improved", dataset_size=750, format='all'):
        """Save evaluation results"""
        
        key = f"{model_name}_{dataset_size}"
        if key not in self.evaluation_results:
            print(f" No evaluation results found for {key}")
            return
        
        results = self.evaluation_results[key]
        print(f" Saving evaluation results for {model_name}...")
        
        # Save summary as CSV
        summary_df = pd.DataFrame([{
            'model_name': results['model_name'],
            'dataset_size': results['dataset_size'],
            'test_samples': results['test_samples'],
            'mean_error': results['metrics']['mean_error'],
            'median_error': results['metrics']['median_error'],
            'std_error': results['metrics']['std_error'],
            'rmse': results['metrics']['rmse'],
            'accuracy_1m': results['accuracy']['accuracy_1m'],
            'accuracy_2m': results['accuracy']['accuracy_2m'],
            'accuracy_3m': results['accuracy']['accuracy_3m']
        }])
        
        csv_path = self.output_dir / f"{model_name}_results_{dataset_size}.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"   CSV summary saved: {csv_path}")
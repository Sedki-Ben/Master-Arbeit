#!/usr/bin/env python3
"""
ResidualCNN- Pipeline Orchestrator

Orchestrates the complete ResidualCNN_training and evaluation pipeline.
ResidualCNN with 
"""

import sys
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from data_loader import ResidualCNNDataLoader
from preprocessing import ResidualCNNPreprocessor
from model import ResidualCNNModel
from training import ResidualCNNTrainer
from evaluation import ResidualCNNEvaluator

class ResidualCNN_ImprovedPipeline:
    """ResidualCNN_ImprovedPipeline."""

    def __init__(self, output_dir="residualcnn_improved_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = ResidualCNNDataLoader()
        self.preprocessor = ResidualCNNPreprocessor()
        self.model_builder = ResidualCNNModel()
        self.trainer = ResidualCNNTrainer(output_dir=output_dir)
        self.evaluator = ResidualCNNEvaluator(output_dir=output_dir)
        
        self.models = {}
        self.results = {}
        
        print(f" ResidualCNN_Pipeline Initialized")
        print(f" Output directory: {self.output_dir}")
    
    def run_complete_pipeline(self, dataset_sizes=[250, 500, 750]):
        """Run complete pipeline for specified dataset sizes"""
        
        print(f" Running ResidualCNN_Pipeline")
        print(f" Dataset sizes: {dataset_sizes}")
        print("=" * 60)
        
        all_results = {}
        
        for dataset_size in dataset_sizes:
            print(f"\n Processing dataset size: {dataset_size}")
            print("-" * 40)
            
            try:
                # Step 1: Load data
                print("1⃣ Loading data...")
                train_data = self._load_single_dataset(dataset_size, "training")
                val_data = self._load_single_dataset(dataset_size, "validation")
                test_data = self._load_single_dataset(750, "testing")
                
                # Step 2: Preprocess data
                print("2⃣ Preprocessing data...")
                processed_train, processed_val, processed_test = self.preprocessor.preprocess_data(
                    train_data, val_data, test_data
                )

                # Step 4: Prepare model inputs
                print("4⃣ Preparing model inputs...")
                X_train = np.stack([processed_train['amplitudes'], processed_train['phases']], axis=-1)
                y_train = processed_train['coordinates']
                X_val = np.stack([processed_val['amplitudes'], processed_val['phases']], axis=-1)
                y_val = processed_val['coordinates']
                X_test = np.stack([processed_test['amplitudes'], processed_test['phases']], axis=-1)
                y_test = processed_test['coordinates']
                
                # Step 5: Build model
                print("5⃣ Building model...")
                model = self.model_builder.build_residualcnn_model(input_shape=(52, 2))
                
                # Step 6: Train model
                print("6⃣ Training model...")
                training_results = self.trainer.compile_and_train_model(
                    model, X_train, y_train, X_val, y_val, dataset_size, "ResidualCNN_Improved"
                )
                
                # Step 7: Evaluate model
                print("7⃣ Evaluating model...")
                evaluation_results = self.evaluator.evaluate_model(
                    model, X_test, y_test, "ResidualCNN_Improved", dataset_size
                )
                
                # Step 8: Generate visualizations
                print("8⃣ Creating visualizations...")
                training_curves_data = {
                    'train_loss': self.trainer.history.history['loss'],
                    'val_loss': self.trainer.history.history['val_loss'],
                    'train_mae': self.trainer.history.history.get('mae', []),
                    'val_mae': self.trainer.history.history.get('val_mae', [])
                }
                self.evaluator.plot_learning_curves(training_curves_data, "ResidualCNN_Improved", dataset_size)
                
                # Step 9: Save results
                print("9⃣ Saving results...")
                self.evaluator.save_results("ResidualCNN_Improved", dataset_size, format='all')
                
                # Store results
                dataset_results = {
                    'training': training_results,
                    'evaluation': evaluation_results,
                    'model': model
                }
                
                all_results[dataset_size] = dataset_results
                self.results[dataset_size] = dataset_results
                self.models[dataset_size] = model
                
                print(f" Dataset size {dataset_size} completed")
                
            except Exception as e:
                print(f" Error processing dataset size {dataset_size}: {e}")
                continue
        
        print(f"\n ResidualCNN_Pipeline Finished!")
        print(f" Successfully processed {len(all_results)}/{len(dataset_sizes)} dataset sizes")
        
        return all_results
    
    def _load_single_dataset(self, dataset_size, point_type):
        """Load single dataset for specified parameters"""
        
        amplitudes, phases, rssi, coordinates = self.data_loader.load_data_by_coordinates(dataset_size, point_type)
        
        return {
            'amplitudes': amplitudes,
            'phases': phases,
            'rssi': rssi,
            'coordinates': coordinates
        }

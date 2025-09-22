#!/usr/bin/env python3
"""
AttentionCNN_Original - Pipeline Orchestrator
===================================================

Orchestrates the complete AttentionCNN_Original training and evaluation pipeline.
AttentionCNN with self-attention mechanism.
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from data_loader import AttentionCNNDataLoader
from preprocessing import AttentionCNNPreprocessor
from model import AttentionCNNModel
from training import AttentionCNNTrainer
from evaluation import AttentionCNNEvaluator

class AttentionCNN_OriginalPipeline:
    """Complete pipeline orchestrator for AttentionCNN_Original model"""
    
    def __init__(self, output_dir="attentioncnn_original_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = AttentionCNNDataLoader()
        self.preprocessor = AttentionCNNPreprocessor()
        self.model_builder = AttentionCNNModel()
        self.trainer = AttentionCNNTrainer(output_dir=output_dir)
        self.evaluator = AttentionCNNEvaluator(output_dir=output_dir)
        
        self.models = {}
        self.results = {}
        
        print(f"🎯 AttentionCNN_Original Complete Pipeline Initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def run_complete_pipeline(self, dataset_sizes=[250, 500, 750]):
        """Run complete pipeline for specified dataset sizes"""
        
        print(f"🚀 Running AttentionCNN_Original Complete Pipeline")
        print(f"📊 Dataset sizes: {dataset_sizes}")
        print("=" * 60)
        
        all_results = {}
        
        for dataset_size in dataset_sizes:
            print(f"\n🔄 Processing dataset size: {dataset_size}")
            print("-" * 40)
            
            try:
                # Step 1: Load data
                print("1️⃣ Loading data...")
                train_data = self._load_single_dataset(dataset_size, "training")
                val_data = self._load_single_dataset(dataset_size, "validation")
                test_data = self._load_single_dataset(750, "testing")
                
                # Step 2: Preprocess data
                print("2️⃣ Preprocessing data...")
                processed_train, processed_val, processed_test = self.preprocessor.preprocess_data(
                    train_data, val_data, test_data
                )

                # Step 3: Apply data augmentation (if original variant)
                print("3️⃣ Applying data augmentation...")
                if hasattr(self.preprocessor, 'apply_data_augmentation'):
                    processed_train = self.preprocessor.apply_data_augmentation(processed_train, augmentation_factor=2)

                # Step 4: Prepare model inputs
                print("4️⃣ Preparing model inputs...")
                X_train = np.stack([processed_train['amplitudes'], processed_train['phases']], axis=-1)
                y_train = processed_train['coordinates']
                X_val = np.stack([processed_val['amplitudes'], processed_val['phases']], axis=-1)
                y_val = processed_val['coordinates']
                X_test = np.stack([processed_test['amplitudes'], processed_test['phases']], axis=-1)
                y_test = processed_test['coordinates']
                
                # Step 5: Build model
                print("5️⃣ Building model...")
                model = self.model_builder.build_attentioncnn_model(input_shape=(52, 2))
                
                # Step 6: Train model
                print("6️⃣ Training model...")
                training_results = self.trainer.compile_and_train_model(
                    model, X_train, y_train, X_val, y_val, dataset_size, "AttentionCNN_Original"
                )
                
                # Step 7: Evaluate model
                print("7️⃣ Evaluating model...")
                evaluation_results = self.evaluator.evaluate_model(
                    model, X_test, y_test, "AttentionCNN_Original", dataset_size
                )
                
                # Step 8: Generate visualizations
                print("8️⃣ Creating visualizations...")
                training_curves_data = {
                    'train_loss': self.trainer.history.history['loss'],
                    'val_loss': self.trainer.history.history['val_loss'],
                    'train_mae': self.trainer.history.history.get('mae', []),
                    'val_mae': self.trainer.history.history.get('val_mae', [])
                }
                self.evaluator.plot_learning_curves(training_curves_data, "AttentionCNN_Original", dataset_size)
                
                # Step 9: Save results
                print("9️⃣ Saving results...")
                self.evaluator.save_results("AttentionCNN_Original", dataset_size, format='all')
                
                # Store results
                dataset_results = {
                    'training': training_results,
                    'evaluation': evaluation_results,
                    'model': model
                }
                
                all_results[dataset_size] = dataset_results
                self.results[dataset_size] = dataset_results
                self.models[dataset_size] = model
                
                print(f"✅ Dataset size {dataset_size} completed successfully")
                
            except Exception as e:
                print(f"❌ Error processing dataset size {dataset_size}: {e}")
                continue
        
        print(f"\n🎉 AttentionCNN_Original Complete Pipeline Finished!")
        print(f"📊 Successfully processed {len(all_results)}/{len(dataset_sizes)} dataset sizes")
        
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

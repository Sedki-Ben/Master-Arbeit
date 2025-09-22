#!/usr/bin/env python3
"""Main script for BasicCNN model training and evaluation"""

import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pipeline import BasicCNN_ImprovedPipeline

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='BasicCNN Indoor Localization Model')
    parser.add_argument('--dataset-sizes', nargs='+', type=int, default=[250, 500, 750],
                       help='Dataset sizes to process (default: 250 500 750)')
    parser.add_argument('--output-dir', type=str, default='basiccnn_improved_results',
                       help='Output directory for results')
    parser.add_argument('--single-size', type=int, default=None,
                       help='Run single experiment with specified dataset size')
    
    args = parser.parse_args()
    
    print("BasicCNN Indoor Localization Model")
    print("=" * 35)
    print(f"Output directory: {args.output_dir}")
    
    pipeline = BasicCNN_ImprovedPipeline(output_dir=args.output_dir)
    
    try:
        if args.single_size:
            print(f"Running single experiment with dataset size: {args.single_size}")
            results = pipeline.run_complete_pipeline([args.single_size])
            
            if args.single_size in results:
                print("Single experiment completed")
            else:
                print("Single experiment failed")
                return 1
        else:
            print(f"Running complete pipeline for dataset sizes: {args.dataset_sizes}")
            results = pipeline.run_complete_pipeline(args.dataset_sizes)
            
            print("pipeline finished successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted")
        return 1
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

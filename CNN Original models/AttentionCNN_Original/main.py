#!/usr/bin/env python3
"""
AttentionCNN_Original- Main 

Main script to run AttentionCNN_Original model training and evaluation.
AttentionCNN with self-attention mechanism.
"""

import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pipeline import AttentionCNN_OriginalPipeline

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='AttentionCNN_Original Indoor Localization Model')
    parser.add_argument('--dataset-sizes', nargs='+', type=int, default=[250, 500, 750],
                       help='Dataset sizes to process (default: 250 500 750)')
    parser.add_argument('--output-dir', type=str, default='attentioncnn_original_results',
                       help='Output directory for results')
    parser.add_argument('--single-size', type=int, default=None,
                       help='Run single experiment with specified dataset size')
    
    args = parser.parse_args()
    
    print(" AttentionCNN_Original Indoor Localization Model")
    print("=" * 45)
    print(f"Output directory: {args.output_dir}")
    
    
    pipeline = AttentionCNN_OriginalPipeline(output_dir=args.output_dir)
    
    try:
        if args.single_size:
            # Run single experiment
            print(f"Running single experiment with dataset size: {args.single_size}")
            results = pipeline.run_complete_pipeline([args.single_size])
            
            if args.single_size in results:
                print(" Single experiment completed")
            else:
                print(" Single experiment failed")
                return 1
        else:
            # Run complete pipeline
            print(f"Running complete pipeline for dataset sizes: {args.dataset_sizes}")
            results = pipeline.run_complete_pipeline(args.dataset_sizes)
            
            print(" pipeline finished successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n Pipeline interrupted")
        return 1
    except Exception as e:
        print(f" Pipeline failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

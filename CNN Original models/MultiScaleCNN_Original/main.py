#!/usr/bin/env python3
"""
MultiScaleCNN_Original - Main Entry Point
===============================================

Main script to run MultiScaleCNN_Original model training and evaluation.
MultiScaleCNN with parallel multi-scale processing.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pipeline import MultiScaleCNN_OriginalPipeline

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='MultiScaleCNN_Original Indoor Localization Model')
    parser.add_argument('--dataset-sizes', nargs='+', type=int, default=[250, 500, 750],
                       help='Dataset sizes to process (default: 250 500 750)')
    parser.add_argument('--output-dir', type=str, default='multiscalecnn_original_results',
                       help='Output directory for results')
    parser.add_argument('--single-size', type=int, default=None,
                       help='Run single experiment with specified dataset size')
    
    args = parser.parse_args()
    
    print("🎯 MultiScaleCNN_Original Indoor Localization Model")
    print("=" * 45)
    print(f"Output directory: {args.output_dir}")
    
    # Initialize pipeline
    pipeline = MultiScaleCNN_OriginalPipeline(output_dir=args.output_dir)
    
    try:
        if args.single_size:
            # Run single experiment
            print(f"Running single experiment with dataset size: {args.single_size}")
            results = pipeline.run_complete_pipeline([args.single_size])
            
            if args.single_size in results:
                print("✅ Single experiment completed successfully")
            else:
                print("❌ Single experiment failed")
                return 1
        else:
            # Run complete pipeline
            print(f"Running complete pipeline for dataset sizes: {args.dataset_sizes}")
            results = pipeline.run_complete_pipeline(args.dataset_sizes)
            
            print("✅ Complete pipeline finished successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

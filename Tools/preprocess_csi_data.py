#!/usr/bin/env python3
"""
CSI Data Preprocessing Tool

Script to preprocess CSI data by:
1. Removing unnecessary columns from raw CSI datasets
2. Reducing CSI data to 104 values (52 amplitude + 52 phase)
3. Cleaning and standardizing the data format
4. Optionally adding RSSI for a total of 105 features
"""

import numpy as np
import pandas as pd
import json
import csv
from pathlib import Path
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

def load_raw_csi_file(file_path: Path) -> pd.DataFrame:
    """Load raw CSI data file and return as DataFrame"""
    
    try:
        # Try to read CSV file
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f" Loaded {len(df)} rows from {file_path.name}")
        return df
    except Exception as e:
        print(f" Error loading {file_path}: {e}")
        return pd.DataFrame()

def identify_necessary_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify necessary and unnecessary columns in the dataset"""
    
    # Define essential columns for CSI-based localization
    essential_columns = {
        'csi_data': ['csi', 'csi_data', 'channel_state_info'],
        'amplitude': ['amplitude', 'amp', 'magnitude'],
        'phase': ['phase', 'ph', 'angle'],
        'rssi': ['rssi', 'signal_strength', 'ss'],
        'coordinates': ['x', 'y', 'coordinate_x', 'coordinate_y', 'pos_x', 'pos_y'],
        'location': ['location', 'position', 'point'],
        'timestamp': ['timestamp', 'time', 'datetime'],
        'antenna': ['antenna', 'ant', 'antenna_id'],
        'subcarrier': ['subcarrier', 'sc', 'carrier'],
        'sample_id': ['sample_id', 'id', 'sample_number']
    }
    
    # Find actual column names in the DataFrame
    found_columns = {}
    for category, possible_names in essential_columns.items():
        found = []
        for col in df.columns:
            if any(name.lower() in col.lower() for name in possible_names):
                found.append(col)
        if found:
            found_columns[category] = found
    
    # Identify unnecessary columns
    all_found = set()
    for cols in found_columns.values():
        all_found.update(cols)
    
    unnecessary_columns = [col for col in df.columns if col not in all_found]
    
    print(f" Column Analysis:")
    print(f"   Essential columns found: {len(all_found)}")
    print(f"   Unnecessary columns: {len(unnecessary_columns)}")
    
    for category, cols in found_columns.items():
        if cols:
            print(f"   {category.upper()}: {cols}")
    
    if unnecessary_columns:
        print(f"   UNNECESSARY: {unnecessary_columns[:10]}{'...' if len(unnecessary_columns) > 10 else ''}")
    
    return {
        'necessary': found_columns,
        'unnecessary': unnecessary_columns
    }

def extract_csi_components(csi_data: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Extract amplitude and phase from CSI data string"""
    
    try:
        # Try to parse as JSON first
        if isinstance(csi_data, str):
            if csi_data.startswith('[') or csi_data.startswith('{'):
                data = json.loads(csi_data)
            else:
                # Try to parse as comma-separated values
                data = [float(x) for x in csi_data.split(',')]
        else:
            data = csi_data
        
        # Handle different CSI data formats
        if isinstance(data, dict):
            # Dictionary format with separate amplitude and phase
            amplitude = data.get('amplitude', data.get('amp', data.get('magnitude')))
            phase = data.get('phase', data.get('ph', data.get('angle')))
        elif isinstance(data, list):
            # List format - assume complex numbers or interleaved real/imaginary
            if len(data) == 104:  # Already separated amplitude and phase
                amplitude = data[:52]
                phase = data[52:]
            elif len(data) == 52:  # Complex numbers as [real, imag, real, imag, ...]
                complex_data = [complex(data[i], data[i+1]) for i in range(0, len(data), 2)]
                amplitude = [abs(c) for c in complex_data]
                phase = [np.angle(c) for c in complex_data]
            else:
                # Assume it's already amplitude or phase data
                amplitude = data[:52] if len(data) >= 52 else data
                phase = None
        else:
            return None, None
        
        # Ensure we have 52 values for each component
        if amplitude and len(amplitude) >= 52:
            amplitude = amplitude[:52]
        if phase and len(phase) >= 52:
            phase = phase[:52]
        
        return amplitude, phase
        
    except Exception as e:
        return None, None

def parse_amplitude_phase_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Parse amplitude and phase from existing columns"""
    
    amplitudes = []
    phases = []
    
    # Look for amplitude column
    amp_col = None
    for col in df.columns:
        if any(name in col.lower() for name in ['amplitude', 'amp', 'magnitude']):
            amp_col = col
            break
    
    # Look for phase column
    phase_col = None
    for col in df.columns:
        if any(name in col.lower() for name in ['phase', 'ph', 'angle']):
            phase_col = col
            break
    
    print(f" Found amplitude column: {amp_col}")
    print(f" Found phase column: {phase_col}")
    
    for idx, row in df.iterrows():
        amp_data = None
        phase_data = None
        
        # Extract amplitude
        if amp_col and pd.notna(row[amp_col]):
            try:
                if isinstance(row[amp_col], str):
                    amp_data = json.loads(row[amp_col])
                else:
                    amp_data = row[amp_col]
                    
                if isinstance(amp_data, list) and len(amp_data) >= 52:
                    amp_data = amp_data[:52]
            except:
                amp_data = None
        
        # Extract phase
        if phase_col and pd.notna(row[phase_col]):
            try:
                if isinstance(row[phase_col], str):
                    phase_data = json.loads(row[phase_col])
                else:
                    phase_data = row[phase_col]
                    
                if isinstance(phase_data, list) and len(phase_data) >= 52:
                    phase_data = phase_data[:52]
            except:
                phase_data = None
        
        # Default to zeros if data is missing
        if amp_data is None:
            amp_data = [0.0] * 52
        if phase_data is None:
            phase_data = [0.0] * 52
        
        amplitudes.append(amp_data)
        phases.append(phase_data)
    
    return np.array(amplitudes), np.array(phases)

def reduce_csi_to_104_values(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce CSI data to exactly 104 values (52 amplitude + 52 phase)"""
    
    print(" Reducing CSI data to 104 values...")
    
    processed_data = []
    
    # Check if we already have amplitude and phase columns
    has_amp_phase = any('amplitude' in col.lower() for col in df.columns) and \
                   any('phase' in col.lower() for col in df.columns)
    
    if has_amp_phase:
        print(" Using existing amplitude and phase columns")
        amplitudes, phases = parse_amplitude_phase_columns(df)
    else:
        # Look for raw CSI data column
        csi_col = None
        for col in df.columns:
            if any(name in col.lower() for name in ['csi', 'channel_state', 'csi_data']):
                csi_col = col
                break
        
        if csi_col:
            print(f" Processing raw CSI data from column: {csi_col}")
            amplitudes = []
            phases = []
            
            for idx, row in df.iterrows():
                if pd.notna(row[csi_col]):
                    amp, phase = extract_csi_components(row[csi_col])
                    if amp and phase:
                        amplitudes.append(amp)
                        phases.append(phase)
                    else:
                        # Use zeros for invalid data
                        amplitudes.append([0.0] * 52)
                        phases.append([0.0] * 52)
                else:
                    amplitudes.append([0.0] * 52)
                    phases.append([0.0] * 52)
            
            amplitudes = np.array(amplitudes)
            phases = np.array(phases)
        else:
            print(" No CSI data columns found!")
            return pd.DataFrame()
    
    # Create new DataFrame with processed data
    result_df = pd.DataFrame()
    
    # Add amplitude columns (52 features)
    for i in range(52):
        result_df[f'amplitude_{i}'] = amplitudes[:, i]
    
    # Add phase columns (52 features)
    for i in range(52):
        result_df[f'phase_{i}'] = phases[:, i]
    
    # Add other essential columns if they exist
    essential_cols = ['x', 'y', 'rssi', 'location', 'timestamp', 'sample_id']
    for col in essential_cols:
        matching_cols = [c for c in df.columns if col.lower() in c.lower()]
        if matching_cols:
            result_df[matching_cols[0]] = df[matching_cols[0]]
    
    print(f" Processed {len(result_df)} samples with {result_df.shape[1]} features")
    print(f"   - Amplitude features: 52")
    print(f"   - Phase features: 52")
    print(f"   - Additional features: {result_df.shape[1] - 104}")
    
    return result_df

def remove_unnecessary_columns(df: pd.DataFrame, keep_columns: List[str] = None) -> pd.DataFrame:
    """Remove unnecessary columns from DataFrame"""
    
    if keep_columns is None:
        # Auto-detect necessary columns
        column_analysis = identify_necessary_columns(df)
        necessary_cols = set()
        for cols in column_analysis['necessary'].values():
            necessary_cols.update(cols)
        keep_columns = list(necessary_cols)
    
    # Filter DataFrame to keep only necessary columns
    available_cols = [col for col in keep_columns if col in df.columns]
    filtered_df = df[available_cols].copy()
    
    removed_count = len(df.columns) - len(available_cols)
    print(f" Removed {removed_count} unnecessary columns")
    print(f" Kept {len(available_cols)} essential columns")
    
    return filtered_df

def validate_processed_data(df: pd.DataFrame) -> Dict[str, any]:
    """Validate the processed data quality"""
    
    validation_results = {
        'total_samples': len(df),
        'total_features': df.shape[1],
        'amplitude_features': len([col for col in df.columns if 'amplitude' in col]),
        'phase_features': len([col for col in df.columns if 'phase' in col]),
        'missing_values': df.isnull().sum().sum(),
        'data_types': df.dtypes.value_counts().to_dict(),
        'amplitude_stats': {},
        'phase_stats': {}
    }
    
    # Analyze amplitude features
    amp_cols = [col for col in df.columns if 'amplitude' in col]
    if amp_cols:
        amp_data = df[amp_cols].values
        validation_results['amplitude_stats'] = {
            'mean': np.mean(amp_data),
            'std': np.std(amp_data),
            'min': np.min(amp_data),
            'max': np.max(amp_data),
            'zeros_count': np.sum(amp_data == 0)
        }
    
    # Analyze phase features
    phase_cols = [col for col in df.columns if 'phase' in col]
    if phase_cols:
        phase_data = df[phase_cols].values
        validation_results['phase_stats'] = {
            'mean': np.mean(phase_data),
            'std': np.std(phase_data),
            'min': np.min(phase_data),
            'max': np.max(phase_data),
            'zeros_count': np.sum(phase_data == 0)
        }
    
    return validation_results

def process_csi_dataset(input_path: Path, output_path: Path, 
                       remove_cols: bool = True, 
                       reduce_csi: bool = True) -> bool:
    """Process a single CSI dataset file"""
    
    print(f"\n Processing: {input_path.name}")
    print("=" * 50)
    
    # Load the data
    df = load_raw_csi_file(input_path)
    if df.empty:
        return False
    
    print(f" Initial data shape: {df.shape}")
    
    # Remove unnecessary columns if requested
    if remove_cols:
        df = remove_unnecessary_columns(df)
        print(f" After column removal: {df.shape}")
    
    # Reduce CSI to 104 values if requested
    if reduce_csi:
        df = reduce_csi_to_104_values(df)
        print(f" After CSI reduction: {df.shape}")
    
    # Validate processed data
    validation = validate_processed_data(df)
    print(f"\n Data Validation:")
    print(f"   Total samples: {validation['total_samples']}")
    print(f"   Total features: {validation['total_features']}")
    print(f"   Amplitude features: {validation['amplitude_features']}")
    print(f"   Phase features: {validation['phase_features']}")
    print(f"   Missing values: {validation['missing_values']}")
    
    if validation['amplitude_stats']:
        amp_stats = validation['amplitude_stats']
        print(f"   Amplitude range: [{amp_stats['min']:.3f}, {amp_stats['max']:.3f}]")
    
    if validation['phase_stats']:
        phase_stats = validation['phase_stats']
        print(f"   Phase range: [{phase_stats['min']:.3f}, {phase_stats['max']:.3f}]")
    
    # Save processed data
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f" Saved processed data to: {output_path}")
        return True
    except Exception as e:
        print(f" Error saving data: {e}")
        return False

def process_multiple_files(input_dir: Path, output_dir: Path, 
                          file_pattern: str = "*.csv",
                          remove_cols: bool = True,
                          reduce_csi: bool = True) -> int:
    """Process multiple CSI dataset files"""
    
    print(f" Processing multiple files from: {input_dir}")
    print(f" Output directory: {output_dir}")
    print(f" File pattern: {file_pattern}")
    
    # Find all matching files
    input_files = list(input_dir.glob(file_pattern))
    
    if not input_files:
        print(f" No files found matching pattern: {file_pattern}")
        return 0
    
    print(f" Found {len(input_files)} files to process")
    
    processed_count = 0
    
    for input_file in input_files:
        # Create output filename
        output_file = output_dir / f"processed_{input_file.name}"
        
        # Process the file
        if process_csi_dataset(input_file, output_file, remove_cols, reduce_csi):
            processed_count += 1
    
    print(f"\n Successfully processed {processed_count}/{len(input_files)} files")
    return processed_count

def create_processing_summary(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create a summary of all processed files"""
    
    print(" Creating processing summary...")
    
    summary_data = []
    
    # Check all processed files
    processed_files = list(output_dir.glob("processed_*.csv"))
    
    for file_path in processed_files:
        try:
            df = pd.read_csv(file_path)
            validation = validate_processed_data(df)
            
            summary_data.append({
                'filename': file_path.name,
                'original_file': file_path.name.replace('processed_', ''),
                'samples': validation['total_samples'],
                'features': validation['total_features'],
                'amplitude_features': validation['amplitude_features'],
                'phase_features': validation['phase_features'],
                'other_features': validation['total_features'] - validation['amplitude_features'] - validation['phase_features'],
                'missing_values': validation['missing_values'],
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            })
        except Exception as e:
            print(f" Error analyzing {file_path}: {e}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f" Processing summary saved: {summary_path}")
        
        # Print summary statistics
        print(f"\n PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {len(summary_data)}")
        print(f"Total samples: {summary_df['samples'].sum()}")
        print(f"Average features per file: {summary_df['features'].mean():.1f}")
        print(f"Total missing values: {summary_df['missing_values'].sum()}")
        print(f"Total output size: {summary_df['file_size_mb'].sum():.2f} MB")
        
        return summary_df
    
    return pd.DataFrame()

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='CSI Data Preprocessing Tool')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file or directory path')
    parser.add_argument('--pattern', type=str, default='*.csv',
                       help='File pattern for batch processing (default: *.csv)')
    parser.add_argument('--keep-columns', action='store_false', dest='remove_cols',
                       help='Keep all columns (do not remove unnecessary ones)')
    parser.add_argument('--no-csi-reduction', action='store_false', dest='reduce_csi',
                       help='Do not reduce CSI to 104 values')
    parser.add_argument('--summary', action='store_true',
                       help='Create processing summary')
    
    args = parser.parse_args()
    
    print(" CSI Data Preprocessing Tool")
    print("=" * 35)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f" Input path does not exist: {input_path}")
        return 1
    
    # Process single file or multiple files
    if input_path.is_file():
        # Single file processing
        success = process_csi_dataset(input_path, output_path, 
                                    args.remove_cols, args.reduce_csi)
        if not success:
            return 1
    else:
        # Multiple files processing
        processed_count = process_multiple_files(input_path, output_path, 
                                               args.pattern, args.remove_cols, 
                                               args.reduce_csi)
        if processed_count == 0:
            return 1
    
    # Create summary if requested
    if args.summary and output_path.is_dir():
        create_processing_summary(input_path, output_path)
    
    print(" CSI data preprocessing complete!")
    return 0

if __name__ == "__main__":
    exit(main())
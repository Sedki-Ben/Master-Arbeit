#!/usr/bin/env python3
"""
Extract Amplitude, Phase, and RSSI Tool
=======================================

Script to extract and format CSI data into the final form:
- 1 RSSI value
- 52 amplitude values (from CSI subcarriers)
- 52 phase values (from CSI subcarriers)

Total: 105 features for machine learning models
"""

import numpy as np
import pandas as pd
import json
import csv
from pathlib import Path
import argparse
import warnings
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')

def parse_json_array(data: Union[str, list]) -> Optional[List[float]]:
    """Parse JSON array from string or return list if already parsed"""
    
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # Try to parse as comma-separated values
            try:
                return [float(x.strip()) for x in data.split(',') if x.strip()]
            except ValueError:
                return None
    elif isinstance(data, list):
        return data
    else:
        return None

def extract_complex_csi_data(csi_data: Union[str, list, dict]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Extract amplitude and phase from complex CSI data"""
    
    try:
        # Parse the data if it's a string
        if isinstance(csi_data, str):
            if csi_data.startswith('[') or csi_data.startswith('{'):
                data = json.loads(csi_data)
            else:
                # Try comma-separated format
                values = [float(x.strip()) for x in csi_data.split(',') if x.strip()]
                data = values
        else:
            data = csi_data
        
        # Handle different CSI data formats
        if isinstance(data, dict):
            # Dictionary format with real and imaginary parts
            if 'real' in data and 'imag' in data:
                real_parts = data['real']
                imag_parts = data['imag']
                if len(real_parts) == len(imag_parts) == 52:
                    complex_values = [complex(r, i) for r, i in zip(real_parts, imag_parts)]
                    amplitudes = [abs(c) for c in complex_values]
                    phases = [np.angle(c) for c in complex_values]
                    return amplitudes, phases
            # Dictionary with amplitude and phase already separated
            elif 'amplitude' in data and 'phase' in data:
                return data['amplitude'][:52], data['phase'][:52]
        
        elif isinstance(data, list):
            if len(data) == 104:
                # Interleaved real and imaginary parts: [r1, i1, r2, i2, ...]
                complex_values = [complex(data[i], data[i+1]) for i in range(0, 104, 2)]
                amplitudes = [abs(c) for c in complex_values]
                phases = [np.angle(c) for c in complex_values]
                return amplitudes, phases
            elif len(data) == 52:
                # Could be complex numbers represented as pairs or just amplitude
                # Assume it's amplitude data
                return data, None
            else:
                # Unknown format, try to use first 52 values
                return data[:52] if len(data) >= 52 else None, None
        
        return None, None
        
    except Exception as e:
        return None, None

def extract_rssi_value(row: pd.Series, rssi_columns: List[str]) -> float:
    """Extract RSSI value from DataFrame row"""
    
    for col in rssi_columns:
        if col in row.index and pd.notna(row[col]):
            try:
                value = float(row[col])
                # Validate RSSI range (typical range: -100 to 0 dBm)
                if -120 <= value <= 10:
                    return value
            except (ValueError, TypeError):
                continue
    
    # Return default RSSI if none found
    return -50.0  # Default RSSI value

def extract_amplitude_values(row: pd.Series, amplitude_columns: List[str]) -> List[float]:
    """Extract amplitude values from DataFrame row"""
    
    for col in amplitude_columns:
        if col in row.index and pd.notna(row[col]):
            # Try to parse amplitude data
            amplitude_data = parse_json_array(row[col])
            if amplitude_data and len(amplitude_data) >= 52:
                return amplitude_data[:52]
    
    # Return zeros if no valid amplitude data found
    return [0.0] * 52

def extract_phase_values(row: pd.Series, phase_columns: List[str]) -> List[float]:
    """Extract phase values from DataFrame row"""
    
    for col in phase_columns:
        if col in row.index and pd.notna(row[col]):
            # Try to parse phase data
            phase_data = parse_json_array(row[col])
            if phase_data and len(phase_data) >= 52:
                return phase_data[:52]
    
    # Return zeros if no valid phase data found
    return [0.0] * 52

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect different types of columns in the DataFrame"""
    
    column_types = {
        'rssi': [],
        'amplitude': [],
        'phase': [],
        'csi_raw': [],
        'coordinates': [],
        'other': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        
        # RSSI columns
        if any(keyword in col_lower for keyword in ['rssi', 'signal_strength', 'ss']):
            column_types['rssi'].append(col)
        
        # Amplitude columns
        elif any(keyword in col_lower for keyword in ['amplitude', 'amp', 'magnitude']):
            column_types['amplitude'].append(col)
        
        # Phase columns
        elif any(keyword in col_lower for keyword in ['phase', 'ph', 'angle']):
            column_types['phase'].append(col)
        
        # Raw CSI data columns
        elif any(keyword in col_lower for keyword in ['csi', 'channel_state', 'csi_data']):
            column_types['csi_raw'].append(col)
        
        # Coordinate columns
        elif any(keyword in col_lower for keyword in ['x', 'y', 'coordinate', 'position', 'location']):
            column_types['coordinates'].append(col)
        
        # Other columns
        else:
            column_types['other'].append(col)
    
    # Print detected columns
    print("📊 Detected Column Types:")
    for col_type, cols in column_types.items():
        if cols:
            print(f"   {col_type.upper()}: {cols}")
    
    return column_types

def extract_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract RSSI, amplitude, and phase features from DataFrame"""
    
    print(f"🔄 Extracting features from {len(df)} samples...")
    
    # Detect column types
    column_types = detect_column_types(df)
    
    # Initialize result data
    result_data = []
    
    for idx, row in df.iterrows():
        # Extract RSSI
        rssi = extract_rssi_value(row, column_types['rssi'])
        
        # Extract amplitude and phase
        amplitude = None
        phase = None
        
        # First, try to extract from separated amplitude/phase columns
        if column_types['amplitude'] and column_types['phase']:
            amplitude = extract_amplitude_values(row, column_types['amplitude'])
            phase = extract_phase_values(row, column_types['phase'])
        
        # If no separated columns, try to extract from raw CSI data
        if (amplitude is None or phase is None) and column_types['csi_raw']:
            for csi_col in column_types['csi_raw']:
                if csi_col in row.index and pd.notna(row[csi_col]):
                    extracted_amp, extracted_phase = extract_complex_csi_data(row[csi_col])
                    if extracted_amp and extracted_phase:
                        amplitude = extracted_amp
                        phase = extracted_phase
                        break
        
        # Use defaults if still None
        if amplitude is None:
            amplitude = [0.0] * 52
        if phase is None:
            phase = [0.0] * 52
        
        # Create feature vector: [rssi, amp_0, ..., amp_51, phase_0, ..., phase_51]
        feature_vector = [rssi] + amplitude + phase
        
        # Add coordinates if available
        sample_data = {'rssi': rssi}
        
        # Add amplitude features
        for i in range(52):
            sample_data[f'amplitude_{i}'] = amplitude[i]
        
        # Add phase features
        for i in range(52):
            sample_data[f'phase_{i}'] = phase[i]
        
        # Add coordinate information if available
        for coord_col in column_types['coordinates']:
            if coord_col in row.index and pd.notna(row[coord_col]):
                sample_data[coord_col] = row[coord_col]
        
        # Add other useful metadata
        for col in ['timestamp', 'sample_id', 'location', 'antenna']:
            matching_cols = [c for c in df.columns if col.lower() in c.lower()]
            if matching_cols and matching_cols[0] in row.index and pd.notna(row[matching_cols[0]]):
                sample_data[matching_cols[0]] = row[matching_cols[0]]
        
        result_data.append(sample_data)
    
    # Create result DataFrame
    result_df = pd.DataFrame(result_data)
    
    print(f"✅ Extracted features:")
    print(f"   - RSSI: 1 feature")
    print(f"   - Amplitude: 52 features")
    print(f"   - Phase: 52 features")
    print(f"   - Additional metadata: {len(result_df.columns) - 105} features")
    
    return result_df

def validate_extracted_features(df: pd.DataFrame) -> Dict[str, any]:
    """Validate the extracted features"""
    
    validation_results = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'rssi_stats': {},
        'amplitude_stats': {},
        'phase_stats': {},
        'missing_values': df.isnull().sum().sum(),
        'feature_ranges': {}
    }
    
    # Validate RSSI
    if 'rssi' in df.columns:
        rssi_data = df['rssi']
        validation_results['rssi_stats'] = {
            'mean': rssi_data.mean(),
            'std': rssi_data.std(),
            'min': rssi_data.min(),
            'max': rssi_data.max(),
            'valid_range': (-120 <= rssi_data.min()) and (rssi_data.max() <= 10)
        }
    
    # Validate amplitude features
    amp_cols = [col for col in df.columns if col.startswith('amplitude_')]
    if amp_cols:
        amp_data = df[amp_cols].values.flatten()
        validation_results['amplitude_stats'] = {
            'count': len(amp_cols),
            'mean': np.mean(amp_data),
            'std': np.std(amp_data),
            'min': np.min(amp_data),
            'max': np.max(amp_data),
            'zeros_count': np.sum(amp_data == 0),
            'non_negative': np.all(amp_data >= 0)  # Amplitudes should be non-negative
        }
    
    # Validate phase features
    phase_cols = [col for col in df.columns if col.startswith('phase_')]
    if phase_cols:
        phase_data = df[phase_cols].values.flatten()
        validation_results['phase_stats'] = {
            'count': len(phase_cols),
            'mean': np.mean(phase_data),
            'std': np.std(phase_data),
            'min': np.min(phase_data),
            'max': np.max(phase_data),
            'zeros_count': np.sum(phase_data == 0),
            'valid_range': (-np.pi <= np.min(phase_data)) and (np.max(phase_data) <= np.pi)  # Phase should be in [-π, π]
        }
    
    return validation_results

def save_extracted_features(df: pd.DataFrame, output_path: Path, format_type: str = 'csv') -> bool:
    """Save extracted features in the specified format"""
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format_type.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format_type.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            print(f"❌ Unsupported format: {format_type}")
            return False
        
        print(f"💾 Saved extracted features to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving features: {e}")
        return False

def create_feature_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a summary of extracted features"""
    
    validation = validate_extracted_features(df)
    
    # Create summary report
    summary_lines = [
        "FEATURE EXTRACTION SUMMARY",
        "=" * 50,
        f"Total samples: {validation['total_samples']}",
        f"Total features: {validation['total_features']}",
        f"Missing values: {validation['missing_values']}",
        "",
        "RSSI STATISTICS:",
        "-" * 20
    ]
    
    if validation['rssi_stats']:
        rssi_stats = validation['rssi_stats']
        summary_lines.extend([
            f"  Mean: {rssi_stats['mean']:.3f} dBm",
            f"  Std:  {rssi_stats['std']:.3f} dBm",
            f"  Range: [{rssi_stats['min']:.3f}, {rssi_stats['max']:.3f}] dBm",
            f"  Valid range: {rssi_stats['valid_range']}"
        ])
    
    summary_lines.extend([
        "",
        "AMPLITUDE STATISTICS:",
        "-" * 25
    ])
    
    if validation['amplitude_stats']:
        amp_stats = validation['amplitude_stats']
        summary_lines.extend([
            f"  Features: {amp_stats['count']}",
            f"  Mean: {amp_stats['mean']:.6f}",
            f"  Std:  {amp_stats['std']:.6f}",
            f"  Range: [{amp_stats['min']:.6f}, {amp_stats['max']:.6f}]",
            f"  Zeros: {amp_stats['zeros_count']} ({amp_stats['zeros_count']/(amp_stats['count']*validation['total_samples'])*100:.1f}%)",
            f"  Non-negative: {amp_stats['non_negative']}"
        ])
    
    summary_lines.extend([
        "",
        "PHASE STATISTICS:",
        "-" * 20
    ])
    
    if validation['phase_stats']:
        phase_stats = validation['phase_stats']
        summary_lines.extend([
            f"  Features: {phase_stats['count']}",
            f"  Mean: {phase_stats['mean']:.6f} rad",
            f"  Std:  {phase_stats['std']:.6f} rad",
            f"  Range: [{phase_stats['min']:.6f}, {phase_stats['max']:.6f}] rad",
            f"  Zeros: {phase_stats['zeros_count']} ({phase_stats['zeros_count']/(phase_stats['count']*validation['total_samples'])*100:.1f}%)",
            f"  Valid range: {phase_stats['valid_range']}"
        ])
    
    # Save summary to file
    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "feature_extraction_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"📊 Feature summary saved: {summary_path}")
    
    # Print summary to console
    print("\n" + summary_text)

def process_single_file(input_path: Path, output_path: Path, format_type: str = 'csv') -> bool:
    """Process a single CSI data file"""
    
    print(f"\n📂 Processing: {input_path.name}")
    print("=" * 50)
    
    try:
        # Load data
        df = pd.read_csv(input_path, encoding='utf-8')
        print(f"📊 Loaded {len(df)} samples from {input_path}")
        
        # Extract features
        result_df = extract_features_from_dataframe(df)
        
        # Validate features
        validation = validate_extracted_features(result_df)
        print(f"\n📈 Validation Results:")
        print(f"   Total features: {validation['total_features']}")
        print(f"   Missing values: {validation['missing_values']}")
        
        # Save results
        return save_extracted_features(result_df, output_path, format_type)
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def process_multiple_files(input_dir: Path, output_dir: Path, 
                          file_pattern: str = "*.csv", 
                          format_type: str = 'csv') -> int:
    """Process multiple CSI data files"""
    
    print(f"🔄 Processing multiple files from: {input_dir}")
    
    # Find all matching files
    input_files = list(input_dir.glob(file_pattern))
    
    if not input_files:
        print(f"❌ No files found matching pattern: {file_pattern}")
        return 0
    
    print(f"📂 Found {len(input_files)} files to process")
    
    processed_count = 0
    all_results = []
    
    for input_file in input_files:
        # Create output filename
        output_file = output_dir / f"extracted_{input_file.stem}.{format_type}"
        
        # Process the file
        if process_single_file(input_file, output_file, format_type):
            processed_count += 1
            
            # Load processed data for combined summary
            try:
                if format_type == 'csv':
                    processed_df = pd.read_csv(output_file)
                    all_results.append(processed_df)
            except:
                pass
    
    # Create combined summary if multiple files were processed
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        create_feature_summary(combined_df, output_dir)
    
    print(f"\n✅ Successfully processed {processed_count}/{len(input_files)} files")
    return processed_count

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Extract Amplitude, Phase, and RSSI Tool')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file or directory path')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'parquet'], 
                       default='csv', help='Output format (default: csv)')
    parser.add_argument('--pattern', type=str, default='*.csv',
                       help='File pattern for batch processing (default: *.csv)')
    parser.add_argument('--summary', action='store_true',
                       help='Create feature extraction summary')
    
    args = parser.parse_args()
    
    print("🔧 Extract Amplitude, Phase, and RSSI Tool")
    print("=" * 45)
    print(f"📊 Target format: 1 RSSI + 52 Amplitude + 52 Phase = 105 features")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        return 1
    
    # Process single file or multiple files
    if input_path.is_file():
        # Single file processing
        success = process_single_file(input_path, output_path, args.format)
        if not success:
            return 1
        
        # Create summary if requested
        if args.summary:
            try:
                if args.format == 'csv':
                    result_df = pd.read_csv(output_path)
                    create_feature_summary(result_df, output_path.parent)
            except Exception as e:
                print(f"⚠️ Could not create summary: {e}")
    else:
        # Multiple files processing
        processed_count = process_multiple_files(input_path, output_path, 
                                               args.pattern, args.format)
        if processed_count == 0:
            return 1
    
    print("✅ Feature extraction complete!")
    return 0

if __name__ == "__main__":
    exit(main())

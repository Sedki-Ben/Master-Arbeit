# CSI Dataset - 750 Samples per Location

## Dataset Information

**Creation Date**: 1756321833.2633955
**Source Dataset**: Amplitude Phase Data Single
**Target Samples per File**: 750
**Sampling Method**: Random sampling with fixed seed (reproducible)

## Statistics

- **Files Processed**: 34
- **Original Total Samples**: 34,238
- **Reduced Total Samples**: 25,500
- **Reduction Percentage**: 25.5%
- **Average Samples per File**: 750

## Purpose

This reduced dataset is designed for:
- Faster training and experimentation
- Memory-constrained environments
- Quick prototyping and parameter tuning
- Computational efficiency studies

## Data Integrity

- Random sampling ensures representative data
- Fixed random seed (42 + file_index) ensures reproducibility
- All original features preserved (RSSI, amplitude array, phase array)
- Spatial coordinates maintained from original filenames

## Usage Notes

- Suitable for CNN model development and testing
- May require adjustments for final production models
- Consider data augmentation to compensate for reduced sample size
- Monitor for potential underfitting due to reduced data volume

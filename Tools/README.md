# Research Tools and Utilities

This directory contains essential tools for data processing, analysis, and visualization in the indoor localization project. These utilities support the entire research pipeline from raw data processing to final result visualization.

## 🛠️ Available Tools

### 📊 Data Processing Tools

#### `extract_amplitude_phase_rssi.py`
**Comprehensive CSI feature extraction tool**
- **Purpose**: Extract RSSI, amplitude, and phase from raw WiFi CSI data
- **Output**: Standardized 105-feature format (1 RSSI + 52 amplitude + 52 phase)
- **Features**: Multi-format support, batch processing, quality validation

```bash
# Extract features from single file
python extract_amplitude_phase_rssi.py --input raw_data.csv --output processed_data.csv

# Batch process entire directory
python extract_amplitude_phase_rssi.py --input "Data/Raw Data/" --output "processed/" --pattern "*.csv"

# Generate processing summary
python extract_amplitude_phase_rssi.py --input data.csv --output result.csv --summary
```

#### `preprocess_csi_data.py`
**Dataset preprocessing and subsampling utility**
- **Purpose**: Create standardized datasets with different sample sizes
- **Features**: Random subsampling, reproducible seeds, quality checks
- **Output**: 250/500/750 sample variants for training efficiency

```bash
# Create 500-sample dataset
python preprocess_csi_data.py --dataset-size 500 --input "original/" --output "CSI Dataset 500/"

# Process multiple sizes
python preprocess_csi_data.py --dataset-sizes 250 500 750
```

#### `phase unwrapping and calibration.py`
**Advanced CSI phase processing**
- **Purpose**: Phase unwrapping and calibration for improved accuracy
- **Features**: Temporal phase unwrapping, calibration algorithms
- **Use case**: Enhanced phase feature quality for better localization

### 📈 Visualization Tools

#### `plot_cnn_original_cdf.py`
**Original CNN models performance visualization**
- **Purpose**: Generate CDF plots for original CNN model comparison
- **Features**: Multi-model comparison, accuracy thresholds, performance tables
- **Output**: High-quality CDF plots and performance summaries

```bash
# Plot CDF for all original models
python plot_cnn_original_cdf.py --models-dir "CNN Original models"

# Generate with performance table
python plot_cnn_original_cdf.py --table --formats png pdf

# Use synthetic data for demonstration
python plot_cnn_original_cdf.py --synthetic
```

#### `plot_cnn_improved_cdf.py`
**Improved CNN models performance visualization**
- **Purpose**: Generate CDF plots for improved CNN model comparison
- **Features**: Enhanced model comparison, improvement analysis
- **Output**: Comparative CDF plots showing improvement over originals

```bash
# Plot improved models CDF
python plot_cnn_improved_cdf.py --models-dir "CNN improved models"

# Compare with original models
python plot_cnn_improved_cdf.py --compare-with-original
```

#### `plot_classical_models_cdf.py`
**Classical methods performance visualization**
- **Purpose**: Generate CDF plots for classical localization methods
- **Features**: k-NN, IDW, and probabilistic model comparison
- **Output**: Classical vs. deep learning comparison plots

```bash
# Plot classical models performance
python plot_classical_models_cdf.py --models-dir "classical models"

# Generate comprehensive comparison
python plot_classical_models_cdf.py --comprehensive
```

#### `plot_cnn_original_learning_curves.py`
**Training progress visualization for original models**
- **Purpose**: Visualize training and validation curves
- **Features**: Loss curves, accuracy progression, overfitting analysis
- **Output**: Learning curve plots for training analysis

#### `plot_cnn_improved_learning_curves.py`
**Training progress visualization for improved models**
- **Purpose**: Compare learning curves between original and improved models
- **Features**: Enhanced training analysis, regularization effects
- **Output**: Comparative learning curve analysis

## 🚀 Quick Start Guide

### 1. Data Processing Pipeline
```bash
# Step 1: Extract features from raw data
cd Tools
python extract_amplitude_phase_rssi.py \
    --input "../Data/Raw Data/Labor Data CSV/" \
    --output "../Data/Processed/" \
    --format csv --summary

# Step 2: Create dataset variants
python preprocess_csi_data.py \
    --dataset-sizes 250 500 750 \
    --input "../Data/Processed/" \
    --output "../Data/"
```

### 2. Model Performance Analysis
```bash
# Generate all CNN performance plots
python plot_cnn_original_cdf.py --table
python plot_cnn_improved_cdf.py --table
python plot_classical_models_cdf.py

# Create learning curve analysis
python plot_cnn_original_learning_curves.py
python plot_cnn_improved_learning_curves.py
```

### 3. Comprehensive Analysis
```bash
# Complete analysis pipeline
for script in plot_*.py; do
    python "$script" --output-dir ../Results/Plots/
done
```

## 📊 Tool Specifications

### Data Processing Capabilities
- **Input Formats**: CSV, JSON, TXT, pickle
- **Output Formats**: CSV, JSON, parquet
- **Batch Processing**: Directory-level operations
- **Quality Control**: Automatic validation and error reporting

### Visualization Features
- **Plot Types**: CDF, scatter plots, learning curves, comparison plots
- **Output Formats**: PNG, PDF, SVG
- **Customization**: Colors, styles, labels, legends
- **Statistics**: Performance tables, summary statistics

### Performance Metrics
- **Localization Error**: Euclidean distance between predicted and true positions
- **Accuracy Thresholds**: 0.5m, 1m, 2m, 3m, 5m accuracy percentages
- **Statistical Measures**: Mean, median, standard deviation, percentiles
- **Comparative Analysis**: Model-to-model performance comparison

## 🔧 Configuration Options

### Common Arguments
```bash
--input PATH           # Input file or directory
--output PATH          # Output file or directory  
--format FORMAT        # Output format (csv, json, png, pdf)
--models-dir PATH      # Directory containing model results
--output-dir PATH      # Directory for output files
--table               # Generate performance summary tables
--synthetic           # Use synthetic data for demonstration
```

### Advanced Options
```bash
--dataset-sizes LIST   # Multiple dataset sizes (250 500 750)
--formats LIST         # Multiple output formats (png pdf svg)
--pattern GLOB         # File pattern for batch processing
--summary             # Generate processing summaries
--compare-with-original # Compare improved vs original models
```

## 📈 Output Examples

### CDF Plots
- **Original Models**: Baseline performance comparison
- **Improved Models**: Enhanced performance with regularization
- **Classical Models**: Traditional method benchmarks
- **Comprehensive**: All methods on single plot

### Performance Tables
```
Model Rankings (750 samples):
Rank  Model              Median Error  1m Accuracy  2m Accuracy
1     ResidualCNN_Improved   0.8m        78%         94%
2     AttentionCNN_Improved  0.9m        75%         92%
3     MultiScaleCNN_Improved 1.0m        72%         90%
...
```

### Learning Curves
- **Training Progress**: Loss and accuracy over epochs
- **Validation Performance**: Overfitting detection
- **Comparison**: Original vs improved training dynamics

## 🎯 Use Case Examples

### Research Workflow
1. **Data Preparation**: Use extraction and preprocessing tools
2. **Model Training**: Train models using processed datasets
3. **Performance Analysis**: Generate CDF plots and tables
4. **Comparison Study**: Compare all model types
5. **Publication**: High-quality figures and comprehensive tables

### Development Workflow
1. **Quick Testing**: Use synthetic data options for tool testing
2. **Incremental Analysis**: Generate plots as models complete training
3. **Debugging**: Use learning curves to identify training issues
4. **Validation**: Compare results across different dataset sizes

## 📋 Dependencies

```python
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0  
matplotlib>=3.5.0
scipy>=1.7.0

# Optional for enhanced features
seaborn>=0.11.0      # Enhanced plotting
plotly>=5.0.0        # Interactive plots
scikit-learn>=1.1.0  # Statistical utilities
```

## 🔍 Troubleshooting

### Common Issues
- **File not found**: Check input paths and file extensions
- **Empty plots**: Verify model results exist and are properly formatted
- **Memory errors**: Use smaller dataset sizes or batch processing
- **Format errors**: Ensure consistent CSV/JSON structure

### Performance Tips
- **Batch Processing**: Use directory-level operations for efficiency
- **Output Formats**: Use PNG for quick viewing, PDF for publications
- **Synthetic Data**: Use for testing tools without running full experiments
- **Parallel Processing**: Process multiple dataset sizes simultaneously

---

**Comprehensive Research Tools for Indoor Localization Analysis**  
*Supporting the complete research pipeline from data to publication*

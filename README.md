# CNN-Based Indoor Localization with WiFi CSI Data

## Project Overview

This Master's thesis project focuses on **deep learning solutions for indoor localization in a singel acssess point scenario** using WiFi Channel State Information (CSI) data and RSSI measurements. The project implements and compares various **CNN architectures** for accurate indoor positioning, evaluating their performance across different dataset sizes and comparing them with classical localization methods.

## Project Structure

```
MasterArbeit/
├── CNN Original models/          # Original CNN architectures
│   ├── BasicCNN_Original/        # Simple 1D CNN baseline
│   ├── HybridCNN_Original/       # CSI + RSSI fusion model
│   ├── AttentionCNN_Original/    # Self-attention mechanism
│   ├── MultiScaleCNN_Original/   # Multi-scale feature extraction
│   └── ResidualCNN_Original/     # Skip connections (ResNet-inspired)
├── CNN improved models/          # Enhanced CNN architectures with regularization
│   ├── BasicCNN_Improved/        # Improved baseline with L2 regularization
│   ├── HybridCNN_Improved/       # Enhanced fusion model
│   ├── AttentionCNN_Improved/    # Improved attention mechanism
│   ├── MultiScaleCNN_Improved/   # Enhanced multi-scale processing
│   └── ResidualCNN_Improved/     # Improved residual connections
├── classical models/             # Classical localization algorithms
│   ├── KNN/                      # k-Nearest Neighbors variants
│   ├── IDW/                      # Inverse Distance Weighting
│   └── Probabilistic/            # Probabilistic fingerprinting
├── Data/                         # CSI datasets and testing points
│   ├── CSI Dataset 250 Samples/  # Reduced dataset for quick experiments
│   ├── CSI Dataset 500 Samples/  # Medium-sized dataset
│   ├── CSI Dataset 750 Samples/  # Full dataset
│   ├── Raw Data/                 # Original laboratory measurements
│   └── Tesing points/           # Evaluation test points
└── Tools/                        # Data processing and visualization tools
```

## CNN Models Implemented

### Original Models
- **BasicCNN**: Simple 1D CNN with 3 convolutional layers and dense layers
- **HybridCNN**: Dual-input model combining CSI and RSSI features
- **AttentionCNN**: Self-attention mechanism for feature importance weighting
- **MultiScaleCNN**: Parallel convolutions with different kernel sizes (3, 7, 15)
- **ResidualCNN**: Skip connections with batch normalization (ResNet-inspired)

### Improved Models
Enhanced versions of all original models featuring:
- **L2 regularization** on all layers
- **Additional dropout layers** for better generalization
- **Improved training strategies** and hyperparameter tuning
- **Better convergence** and reduced overfitting

## Dataset Information

### CSI Data Features
- **Input Shape**: (52, 2) - 52 subcarriers × 2 features (amplitude + phase)
- **Additional Features**: RSSI values for hybrid models
- **Coordinate System**: 2D indoor positioning (X, Y coordinates)
- **Data Collection**: Laboratory environment with controlled measurements

### Dataset Variants
- **250 Samples**: Quick experimentation and prototyping
- **500 Samples**: Balanced performance vs. computation time
- **750 Samples**: Full dataset for optimal performance

### Spatial Coverage
- **Grid Points**: Multiple measurement locations in indoor environment include training and validation points
- **Test Points**: Independent evaluation points for realistic assessment
- **Coordinate Range**: Covers typical indoor room dimensions

## Quick Start

### Run CNN Models

#### Original Models
```bash
# Run all original CNN models
cd "CNN Original models/BasicCNN_Original"
python main.py --dataset-sizes 250 500 750

# Run specific model with single dataset size
cd "CNN Original models/AttentionCNN_Original"
python main.py --single-size 500
```

#### Improved Models
```bash
# Run improved CNN models
cd "CNN improved models/BasicCNN_Improved"
python main.py --dataset-sizes 250 500 750

# Run all improved models comparison
cd "CNN improved models"
# Each model directory contains similar structure
```

### Run Classical Models
```bash
cd "classical models"
python main.py --mode comprehensive  # All classical methods
python main.py --mode quick          # Quick comparison
```

### Data Processing
```bash
cd Tools
python extract_amplitude_phase_rssi.py --input "Data/Raw Data" --output "processed"
python preprocess_csi_data.py --dataset-size 500
```

### Generate Visualizations
```bash
cd Tools
python plot_cnn_original_cdf.py
python plot_cnn_improved_cdf.py
python plot_classical_models_cdf.py
```

## Performance Metrics

All models are evaluated using:
- **Localization Error**: Euclidean distance between predicted and true positions: median, mean.
- **Accuracy Thresholds**: Percentage of predictions within 0.5m, 1m, 2m, 3m,
- **Cumulative Distribution Function (CDF)**: Complete error distribution analysis

## Requirements

```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
scipy>=1.7.0
```

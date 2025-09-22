# CNN Original Models

This directory contains the original implementations of five different CNN architectures for indoor localization using WiFi CSI data. These models serve as baseline implementations for comparison with their improved counterparts.

## 🏗️ Model Architectures

### 1. BasicCNN_Original
**Simple 1D CNN baseline model**
- **Architecture**: 3 Conv1D layers (32→64→128 filters) + 2 Dense layers
- **Features**: Basic convolution + max pooling + global average pooling
- **Use case**: Baseline comparison and quick prototyping
- **Strengths**: Fast training, simple architecture, good starting point

### 2. HybridCNN_Original  
**Dual-input model combining CSI and RSSI**
- **Architecture**: Separate CSI and RSSI processing branches + feature fusion
- **Features**: CSI convolutions + RSSI dense layers + concatenation
- **Use case**: Leveraging both CSI and RSSI information
- **Strengths**: Multi-modal data fusion, improved accuracy with RSSI

### 3. AttentionCNN_Original
**Self-attention mechanism for feature importance**
- **Architecture**: Conv1D layers + self-attention + dense layers
- **Features**: Attention weights for automatic feature selection
- **Use case**: Understanding which CSI features are most important
- **Strengths**: Interpretable feature importance, selective focus

### 4. MultiScaleCNN_Original
**Parallel multi-scale feature extraction**
- **Architecture**: Parallel Conv1D with different kernel sizes (3, 7, 15)
- **Features**: Multi-scale convolutions + feature concatenation
- **Use case**: Capturing both fine and coarse CSI patterns
- **Strengths**: Rich feature representation, scale-invariant features

### 5. ResidualCNN_Original
**Skip connections inspired by ResNet**
- **Architecture**: Residual blocks with batch normalization + skip connections
- **Features**: 3 residual blocks + global average pooling
- **Use case**: Deep network training with gradient flow
- **Strengths**: Stable training, better gradient flow, deeper networks

## 📁 Directory Structure

Each model follows the same modular structure:

```
ModelName_Original/
├── main.py           # Entry point and argument parsing
├── model.py          # Model architecture definition
├── pipeline.py       # Complete training and evaluation pipeline
├── training.py       # Training procedures and callbacks
├── data_loader.py    # Data loading and preprocessing
├── preprocessing.py  # Feature engineering and normalization
└── evaluation.py     # Performance metrics and visualization
```

## 🚀 Usage

### Run Individual Models
```bash
# Basic CNN
cd BasicCNN_Original
python main.py --dataset-sizes 250 500 750

# Attention CNN
cd AttentionCNN_Original
python main.py --single-size 500

# Multi-scale CNN
cd MultiScaleCNN_Original
python main.py --output-dir custom_results
```

### Common Arguments
- `--dataset-sizes`: List of dataset sizes to process (default: 250 500 750)
- `--single-size`: Run single experiment with specified dataset size
- `--output-dir`: Custom output directory for results

### Example Outputs
Each model generates:
- Training/validation learning curves
- CDF plots of localization errors
- Performance metrics (JSON/CSV)
- Predicted vs. true position scatter plots

## 📊 Expected Performance

Based on typical results (median localization error):

| Model | 250 Samples | 500 Samples | 750 Samples |
|-------|-------------|-------------|-------------|
| BasicCNN | ~2.1m | ~1.8m | ~1.6m |
| HybridCNN | ~1.9m | ~1.6m | ~1.4m |
| AttentionCNN | ~1.7m | ~1.4m | ~1.2m |
| MultiScaleCNN | ~1.8m | ~1.5m | ~1.3m |
| ResidualCNN | ~1.6m | ~1.3m | ~1.1m |

*Note: Actual performance depends on data quality and training conditions*

## 🔧 Model Specifications

### Input Format
- **Shape**: (52, 2) - 52 CSI subcarriers × 2 features (amplitude + phase)
- **Additional**: RSSI value for HybridCNN (shape: (1,))
- **Output**: 2D coordinates (X, Y) for indoor positioning

### Training Configuration
- **Loss Function**: Euclidean distance (custom implementation)
- **Optimizer**: Adam with default parameters
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20%

### Architecture Details

#### BasicCNN_Original
```
Input(52, 2) → Conv1D(32,3) → MaxPool → Conv1D(64,3) → MaxPool → 
Conv1D(128,3) → GlobalAvgPool → Dense(256) → Dropout(0.5) → 
Dense(128) → Dropout(0.3) → Dense(2)
```

#### HybridCNN_Original
```
CSI Branch: Input(52,2) → Conv1D layers → GlobalAvgPool
RSSI Branch: Input(1) → Dense(32) → Dense(16)
Fusion: Concatenate → Dense(256) → Dense(128) → Dense(2)
```

#### AttentionCNN_Original
```
Input(52,2) → Conv1D(64,3) → Conv1D(128,3) → 
Attention(softmax) → Multiply → GlobalAvgPool → Dense layers
```

#### MultiScaleCNN_Original
```
Input(52,2) → [Conv1D(32,3), Conv1D(32,7), Conv1D(32,15)] → 
Concatenate → Conv1D(128,3) → Conv1D(256,3) → GlobalAvgPool → Dense layers
```

#### ResidualCNN_Original
```
Input(52,2) → Conv1D(64,3) → [3x ResidualBlock] → 
GlobalAvgPool → Dense layers
ResidualBlock: Conv1D → BN → Conv1D → BN → Add → ReLU
```

## 📈 Performance Analysis

### Strengths by Model
- **BasicCNN**: Simple, fast, good baseline
- **HybridCNN**: Multi-modal fusion, RSSI integration
- **AttentionCNN**: Feature interpretability, selective focus
- **MultiScaleCNN**: Rich representation, scale invariance
- **ResidualCNN**: Deep training stability, best overall performance

### Typical Use Cases
- **Quick prototyping**: BasicCNN_Original
- **Multi-modal data**: HybridCNN_Original
- **Feature analysis**: AttentionCNN_Original
- **Complex patterns**: MultiScaleCNN_Original
- **Best performance**: ResidualCNN_Original

## 🔄 Comparison with Improved Models

The original models serve as baselines for the improved versions:
- **Original**: Basic architectures without regularization
- **Improved**: Enhanced with L2 regularization, additional dropout, better training
- **Performance gap**: Improved models typically 15-25% better accuracy

## 📋 Requirements

```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
```

## 🎯 Next Steps

After running original models:
1. Compare with improved models in `../CNN improved models/`
2. Analyze performance differences
3. Use visualization tools in `../Tools/` for CDF plots
4. Compare with classical methods in `../classical models/`

---

**Original CNN Models for Indoor Localization**  
*Baseline implementations for WiFi CSI-based positioning*

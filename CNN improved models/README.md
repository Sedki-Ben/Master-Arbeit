# CNN Improved Models

This directory contains enhanced versions of the original CNN architectures with advanced regularization techniques, improved training strategies, and optimized hyperparameters. These models demonstrate significant performance improvements over their original counterparts.

## Key Improvements

### Enhanced Regularization
- **L2 Regularization**: Applied to all convolutional and dense layers (1e-4)
- **Additional Dropout**: Strategic placement for better generalization
- **Improved Architecture**: Optimized layer configurations

### Training Enhancements
- **Better Convergence**: Improved training stability
- **Reduced Overfitting**: Solved data leakage, more robust to unseen data
- **Faster Training**: Optimized hyperparameters during training
- **Performance Boost**: accuracy improvement over original models in some cases while no noticeable improvement in others.

##  Enhanced Model Architectures

All the following architectures were trained after changing the training parameters, such as batch size, learning rate to match the model and the training data size at each iteration.

### 1. BasicCNN_Improved
**Enhanced baseline with comprehensive regularization**
- **Improvements**: L2 regularization on all layers + additional dropout layers
- **Architecture**: 3 Conv1D layers with regularization + enhanced dense layers
- **Performance**: ~20% better than original BasicCNN

### 2. HybridCNN_Improved
**Advanced dual-input model with enhanced fusion**
- **Improvements**: Regularized CSI and RSSI branches + improved fusion strategy
- **Architecture**: Enhanced feature extraction + optimized combination
- **Performance**: Significant accuracy improvement yielding the best results in this work

### 3. AttentionCNN_Improved
**Advanced attention mechanism with regularization**
- **Improvements**: Regularized attention weights + enhanced feature selection
- **Architecture**: Improved attention layers + regularized convolutions
- **Performance**: no noticeable improvement

### 4. MultiScaleCNN_Improved
**Enhanced multi-scale processing with regularization**
- **Improvements**: Regularized multi-scale branches + improved feature fusion
- **Architecture**: Enhanced parallel convolutions + optimized concatenation
- **Performance**: no significant improvement

### 5. ResidualCNN_Improved
**Advanced residual connections with comprehensive regularization**
- **Improvements**: Regularized residual blocks + enhanced skip connections
- **Architecture**: Improved gradient flow + optimized depth
- **Performance**: deterioration of already low accuracy, shallow networks are more suitable to our problem setting


## Directory Structure

Each improved model maintains the same modular structure as originals:

```
ModelName_Improved/
├── main.py           # Entry point with enhanced arguments
├── model.py          # Improved model architecture
├── pipeline.py       # Enhanced training and evaluation pipeline
├── training.py       # Advanced training procedures
├── data_loader.py    # Optimized data loading
├── preprocessing.py  # Enhanced feature engineering
└── evaluation.py     # Comprehensive performance metrics
```

## Usage

### Run Individual Improved Models
```bash
# Enhanced Basic CNN
cd BasicCNN_Improved
python main.py --dataset-sizes 250 500 750

# Enhanced Attention CNN with single dataset
cd AttentionCNN_Improved
python main.py --single-size 500

# Enhanced Multi-scale CNN with custom output
cd MultiScaleCNN_Improved
python main.py --output-dir enhanced_results
```

### Advanced Training Options
```bash
# All models support the same interface as originals
python main.py --dataset-sizes 250 500 750 --output-dir results
python main.py --single-size 750  # Best performance with largest dataset
```

## Performance Improvements

reminder of the rsults from the original models (median localization error):

| Model | 250 Samples | 500 Samples | 750 Samples |
|-------|-------------|-------------|-------------|
| BasicCNN | ~2.5m | ~2.46m | ~2.23m |
| HybridCNN | ~1.61m | ~1.8m | ~1.91m |
| AttentionCNN | ~2.24m | ~2.08m | ~1.82m |
| MultiScaleCNN | ~2.6m | ~2.6m | ~2.5m |
| ResidualCNN | ~2.34m | ~2.29m | ~2.21m |

### Acresults of improved setting

| Model | 250 Samples | 500 Samples | 750 Samples |
|-------|-------------|-------------|-------------|
| BasicCNN | ~1.68m | ~1.85m | ~1.82m |
| HybridCNN | ~1.14m | ~1.34m | ~1.6m |
| AttentionCNN | ~2.1m | ~1.92m | ~1.77m |
| MultiScaleCNN | ~2.34m | ~2.4m | ~2.42m |
| ResidualCNN | ~2.41m | ~2.7m | ~2.72m |

## Technical Enhancements

### Regularization Strategy
```python
# L2 Regularization on all layers
kernel_regularizer=regularizers.l2(1e-4)

# Strategic Dropout Placement
layers.Dropout(0.3)  # After conv layers
layers.Dropout(0.4)  # After global pooling
layers.Dropout(0.5)  # After first dense layer
```

### Enhanced Architecture Components
- **Convolutional Layers**: All include L2 regularization
- **Dropout Layers**: Additional strategic placement
- **Dense Layers**: Regularized with optimized dropout rates
- **Global Pooling**: Enhanced with dropout for better generalization

### Training Improvements
- **Learning Rate**: Optimized for each architecture
- **Batch Size**: Fine-tuned for better convergence
- **Early Stopping**: Improved patience and monitoring
- **Validation Strategy**: Enhanced cross-validation approach

## Architecture Specifications

### BasicCNN_Improved
```
Input(52,2) → Conv1D(32,3)+L2+Dropout(0.3) → MaxPool → 
Conv1D(64,3)+L2+Dropout(0.3) → MaxPool → 
Conv1D(128,3)+L2 → GlobalAvgPool+Dropout(0.4) → 
Dense(256)+L2+Dropout(0.5) → Dense(128)+L2+Dropout(0.4) → Dense(2)
```

### HybridCNN_Improved
```
CSI Branch: Enhanced conv layers with regularization
RSSI Branch: Regularized dense layers
Fusion: Improved concatenation + regularized dense layers
```

### Enhanced Training Features
- **Regularization**: Comprehensive L2 + dropout strategy
- **Stability**: Improved convergence with better hyperparameters
- **Generalization**: Reduced overfitting through enhanced regularization
- **Performance**: improvement over originals for both basic CNN and Hybrid CNN, indicating better performance of simpler CNN-based architectures. the localization performance in our case faced significan bottlenecks due to limited data restricted by the deliberate use of commodity hardware. 






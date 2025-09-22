# CNN-Based Indoor Localization with WiFi CSI Data

## 🎯 Project Overview

This Master's thesis project focuses on **deep learning solutions for indoor localization** using WiFi Channel State Information (CSI) data. The project implements and compares various **CNN architectures** for accurate indoor positioning, evaluating their performance across different dataset sizes and comparing them with classical localization methods.

## 🏗️ Project Structure

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

## 🧠 CNN Models Implemented

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

## 📊 Dataset Information

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
- **Grid Points**: Multiple measurement locations in indoor environment
- **Test Points**: Independent evaluation points for realistic assessment
- **Coordinate Range**: Covers typical indoor room dimensions

## 🚀 Quick Start

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

## 📈 Performance Metrics

All models are evaluated using:
- **Localization Error**: Euclidean distance between predicted and true positions
- **Accuracy Thresholds**: Percentage of predictions within 0.5m, 1m, 2m, 3m, 5m
- **Statistical Measures**: Mean, median, standard deviation, percentiles
- **Cumulative Distribution Function (CDF)**: Complete error distribution analysis

## 🔬 Research Contributions

### 1. Comprehensive CNN Architecture Comparison
- Systematic evaluation of 5 different CNN architectures
- Original vs. improved model variants
- Performance analysis across different dataset sizes

### 2. Classical vs. Deep Learning Comparison
- Baseline comparison with traditional methods (k-NN, IDW, Probabilistic)
- Quantitative analysis of deep learning advantages
- Computational efficiency trade-offs

### 3. Dataset Size Impact Analysis
- Performance scaling with different training data sizes
- Optimization strategies for limited data scenarios
- Practical deployment considerations

### 4. Feature Engineering and Processing
- CSI amplitude and phase feature extraction
- RSSI integration for hybrid models
- Data preprocessing and normalization strategies

## 🛠️ Technical Stack

- **Deep Learning**: TensorFlow/Keras
- **Classical ML**: scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Spatial Analysis**: SciPy

## 📋 Requirements

```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
scipy>=1.7.0
```

## 🎯 Key Results

### CNN Model Performance Hierarchy
1. **ResidualCNN** - Best overall performance with skip connections
2. **AttentionCNN** - Excellent feature selection capabilities
3. **MultiScaleCNN** - Strong multi-scale feature extraction
4. **HybridCNN** - Good CSI+RSSI fusion performance
5. **BasicCNN** - Solid baseline performance

### Dataset Size Impact
- **750 samples**: Optimal performance across all models
- **500 samples**: Good balance of performance vs. training time
- **250 samples**: Suitable for quick prototyping and testing

### Deep Learning vs. Classical Methods
- **CNN models** consistently outperform classical methods
- **Improved models** show 15-25% better accuracy than original versions
- **Attention mechanisms** provide best feature interpretability

## 📚 Publication and Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{sedki2024cnn_indoor_localization,
  title={CNN-Based Indoor Localization with WiFi CSI Data},
  author={Sedki Ben [Your Name]},
  year={2024},
  school={[Your University]},
  type={Master's Thesis}
}
```

## 🤝 Contributing

This project is part of a Master's thesis research. For questions or collaborations:
- Create an issue for bug reports or feature requests
- Fork the repository for improvements
- Contact the author for research collaborations

## 📄 License

This project is available for academic and research purposes. Please cite appropriately when using this work.

## 🔗 Repository

GitHub: [https://github.com/Sedki-Ben/Master-Arbeit](https://github.com/Sedki-Ben/Master-Arbeit)

---

**Master's Thesis Project - CNN-Based Indoor Localization**  
*Deep Learning Solutions for WiFi CSI-Based Indoor Positioning*

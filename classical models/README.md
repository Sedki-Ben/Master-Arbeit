# Classical Models for Indoor Localization

This directory contains a modular implementation of classical localization algorithms for indoor positioning using CSI (Channel State Information) data.

## Directory Structure

```
classical models/
├── shared/                 # Shared utilities for all models
│   ├── data_loader.py     # Data loading functions
│   ├── preprocessing.py   # Feature preprocessing utilities
│   └── evaluation.py      # Performance evaluation and visualization
├── KNN/                   # k-Nearest Neighbors models
│   ├── model.py          # k-NN implementations
│   ├── pipeline.py       # Complete k-NN evaluation pipeline
│   └── main.py           # k-NN entry point
├── IDW/                   # Inverse Distance Weighting models
│   ├── model.py          # IDW implementations
│   ├── pipeline.py       # Complete IDW evaluation pipeline
│   └── main.py           # IDW entry point
├── Probabilistic/         # Probabilistic fingerprinting models
│   ├── model.py          # Probabilistic implementations
│   ├── pipeline.py       # Complete probabilistic evaluation pipeline
│   └── main.py           # Probabilistic entry point
├── main.py               # Main entry point for all models
└── README.md             # This file
```

## Models Implemented

### 1. k-Nearest Neighbors (k-NN)
- **Basic k-NN**: Standard k-NN with different k values
- **Distance-weighted k-NN**: Uses inverse distance weighting
- **k-NN Ensemble**: Combines multiple k values
- **Multi-k-NN**: Ensemble with different combination strategies

### 2. Inverse Distance Weighting (IDW)
- **Basic IDW**: Standard IDW with different power parameters
- **Adaptive IDW**: Locally adaptive power parameters
- **Multi-distance metrics**: Euclidean, Manhattan, Minkowski
- **Neighbor-limited IDW**: Uses only k nearest neighbors
- **IDW Ensemble**: Combines multiple power values

### 3. Probabilistic Fingerprinting
- **Basic Probabilistic**: Gaussian distributions with MLE
- **Advanced Covariance**: Ledoit-Wolf, OAS, Shrunk estimators
- **Gaussian Mixture Models**: Multi-component distributions
- **Bayesian Models**: With different prior types
- **Regularization Analysis**: Different smoothing parameters

## Quick Start

### Run All Models (Comprehensive Comparison)
```bash
cd "classical models"
python main.py --mode comprehensive
```

### Run Individual Model Types
```bash
# k-NN models only
python main.py --mode knn

# IDW models only
python main.py --mode idw

# Probabilistic models only
python main.py --mode probabilistic

# Quick comparison (basic models)
python main.py --mode quick
```

### Run Specific Model Categories
```bash
# k-NN with different configurations
cd KNN
python main.py --mode basic        # Basic k-NN evaluation
python main.py --mode enhanced     # With statistical features
python main.py --mode comparison   # Basic vs Enhanced
python main.py --mode quick        # Quick test

# IDW with different configurations
cd IDW
python main.py --mode basic        # Basic IDW evaluation
python main.py --mode power        # Power parameter analysis
python main.py --mode adaptive     # Adaptive vs Standard
python main.py --mode quick        # Quick test

# Probabilistic with different configurations
cd Probabilistic
python main.py --mode basic        # Basic probabilistic
python main.py --mode covariance   # Covariance estimator analysis
python main.py --mode gmm          # Gaussian Mixture Models
python main.py --mode bayesian     # Bayesian comparison
python main.py --mode quick        # Quick test
```

## Features

### Data Loading (`shared/data_loader.py`)
- **Multiple data sources**: Amplitude Phase Data Single, CSI Dataset folders
- **Coordinate extraction**: Automatically extracts coordinates from filenames
- **Train/test splitting**: Location-based stratified splitting
- **Feature combination**: Amplitude (52 subcarriers) + RSSI (1 value) = 53 features

### Preprocessing (`shared/preprocessing.py`)
- **Multiple scalers**: StandardScaler, MinMaxScaler, RobustScaler
- **Statistical features**: Mean, std, max, min, range, median, quartiles, energy, RMS
- **Outlier removal**: IQR and Z-score methods
- **Feature extraction**: Amplitude-only, RSSI-only, or combined features

### Evaluation (`shared/evaluation.py`)
- **Comprehensive metrics**: Mean/median error, accuracy at multiple thresholds
- **Visualizations**: CDF plots, scatter plots, error distributions
- **Model comparison**: Multi-model CDF comparison
- **Results export**: CSV and JSON formats

## Model Configurations

### k-NN Models
- **k values**: 1, 3, 5, 7, 9
- **Distance metrics**: Euclidean, Manhattan, Minkowski
- **Weighting**: Uniform, distance-weighted
- **Ensemble methods**: Average, weighted average, median

### IDW Models
- **Power values**: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0
- **Distance metrics**: Euclidean, Manhattan, Minkowski
- **Adaptive methods**: Density-based, variance-based, distance-based
- **Neighbor limits**: 10, 20, 50, 100, unlimited

### Probabilistic Models
- **Covariance estimators**: Empirical, Ledoit-Wolf, OAS, Shrunk
- **Smoothing parameters**: 1e-8 to 1e-2
- **GMM components**: 1, 2, 3 components
- **GMM covariance types**: Full, diagonal, tied, spherical
- **Bayesian priors**: Uniform, distance-based, density-based

## Performance Metrics

All models are evaluated using:
- **Localization error**: Euclidean distance between true and predicted positions
- **Accuracy thresholds**: 0.5m, 1m, 2m, 3m, 5m
- **Statistical measures**: Mean, median, standard deviation, percentiles
- **Coordinate-wise analysis**: Separate X and Y coordinate errors

## Visualizations

Each model generates:
- **CDF plots**: Cumulative distribution of localization errors
- **Scatter plots**: True vs predicted positions
- **Comparison plots**: Multi-model performance comparison
- **Learning analysis**: Model-specific performance insights

## Output Structure

Results are saved in organized directories:
```
{model_type}_results/
├── {model_name}_results.json           # Individual model results
├── {model_name}_error_distribution.png # CDF plot
├── {model_name}_predictions_scatter.png # Scatter plot
├── classical_models_cdf_comparison.png  # Multi-model comparison
├── classical_models_performance_summary.csv # Summary table
└── {model_type}_complete_summary.json   # Complete evaluation summary
```

## Advanced Features

### Ensemble Methods
- **k-NN Ensemble**: Multiple k values with different combination strategies
- **IDW Ensemble**: Multiple power parameters with averaging/median
- **Covariance Ensemble**: Multiple covariance estimators

### Adaptive Algorithms
- **Adaptive IDW**: Power parameter varies based on local data characteristics
- **Bayesian Inference**: Incorporates spatial priors about reference point likelihood

### Statistical Features
- **Amplitude statistics**: Mean, std, range, quartiles, energy
- **Spectral features**: Based on subcarrier variations
- **Energy features**: RMS, total energy across subcarriers

## Testing and Development

### Quick Testing
Use `--mode quick` for fast testing during development:
```bash
python main.py --mode quick  # Tests basic versions of all models
```

### Individual Component Testing
```bash
# Test specific components
cd KNN && python main.py --mode quick
cd IDW && python main.py --mode quick
cd Probabilistic && python main.py --mode quick
```

## Expected Performance Hierarchy

Based on typical indoor localization results:
1. **Probabilistic models** (especially with proper covariance estimation)
2. **k-NN models** (with optimal k and distance weighting)
3. **IDW models** (with optimal power parameter)

However, performance depends on:
- Data quality and quantity
- Reference point density
- Environmental characteristics
- Feature preprocessing quality

## Customization

### Adding New Models
1. Create model class in appropriate `model.py`
2. Add evaluation method in `pipeline.py`
3. Update `main.py` argument parsing if needed

### Modifying Features
- Edit `shared/preprocessing.py` for new feature engineering
- Update `shared/data_loader.py` for new data sources
- Modify `shared/evaluation.py` for new metrics

## Dependencies

- **NumPy**: Numerical computations
- **Pandas**: Data handling
- **Scikit-learn**: Machine learning utilities, preprocessing, covariance estimation
- **SciPy**: Statistical functions, spatial distance calculations
- **Matplotlib**: Visualization and plotting


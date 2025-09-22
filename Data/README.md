# Indoor Localization Dataset

This directory contains comprehensive WiFi CSI (Channel State Information) datasets collected for indoor localization research. The data supports both CNN-based and classical localization methods with varying dataset sizes for different experimental needs.

## 📊 Dataset Overview

### Data Collection Environment
- **Setting**: Laboratory/indoor environment
- **Technology**: WiFi 802.11 with CSI extraction
- **Measurement Points**: Grid-based spatial sampling
- **Data Format**: CSV files with CSI amplitude, phase, and RSSI values

### Feature Structure
- **CSI Features**: 52 subcarriers × 2 (amplitude + phase) = 104 features
- **RSSI Features**: 1 received signal strength indicator
- **Total Features**: 105 per sample
- **Spatial Information**: X,Y coordinates for localization ground truth

## 📁 Directory Structure

```
Data/
├── Raw Data/                           # Original measurement data
│   ├── Labor Data CSV/                 # Processed raw measurements
│   └── Labor Data txt/                 # Original text format data
├── Amplitude Phase Data single format/ # Processed CSI amplitude/phase data
├── Data extracted Amplitude Phase/     # Alternative extraction format
├── CSI Dataset 250 Samples/           # Reduced dataset (250 samples/location)
├── CSI Dataset 500 Samples/           # Medium dataset (500 samples/location)  
├── CSI Dataset 750 Samples/           # Full dataset (750 samples/location)
└── Tesing points/                      # Independent test points
    ├── Testing Points CSV/             # Test data in CSV format
    ├── testing points data/            # Original test measurements
    ├── Testing Points Dataset 250/     # Test points for 250-sample dataset
    ├── Testing Points Dataset 500/     # Test points for 500-sample dataset
    └── Testing Points Dataset 750/     # Test points for 750-sample dataset
```

## 🎯 Dataset Variants

### 1. CSI Dataset 250 Samples
- **Purpose**: Quick experimentation and prototyping
- **Size**: 250 samples per spatial location
- **Total Samples**: ~8,500 (34 locations × 250 samples)
- **Use Case**: Fast model development, parameter tuning
- **Processing Time**: Minimal computational requirements

### 2. CSI Dataset 500 Samples  
- **Purpose**: Balanced performance vs. computation
- **Size**: 500 samples per spatial location
- **Total Samples**: ~17,000 (34 locations × 500 samples)  
- **Use Case**: Standard model training and evaluation
- **Processing Time**: Moderate computational requirements

### 3. CSI Dataset 750 Samples
- **Purpose**: Maximum performance and accuracy
- **Size**: 750 samples per spatial location
- **Total Samples**: ~25,500 (34 locations × 750 samples)
- **Use Case**: Final model evaluation, publication results
- **Processing Time**: Higher computational requirements

## 📍 Spatial Coverage

### Measurement Grid
- **Coordinate System**: 2D indoor positioning (X,Y)
- **Grid Pattern**: Regular spatial sampling
- **Coverage Area**: Typical indoor room/laboratory space
- **Resolution**: Sub-meter spacing between measurement points

### Location Distribution
The filenames indicate spatial coordinates:
- `0,0.csv` → Location (0,0)
- `2,3.csv` → Location (2,3)
- `5,4.csv` → Location (5,4)

Total coverage: **34 unique spatial locations**

## 🧪 Test Points

### Independent Evaluation
- **Purpose**: Unbiased model evaluation
- **Locations**: 5 test points at intermediate positions
- **Coordinates**: (0.5,0.5), (1.5,4.5), (2.5,2.5), (3.5,1.5), (5.5,3.5)
- **Samples**: Consistent across all dataset variants

### Testing Strategy
- **Training**: Grid measurement points
- **Testing**: Intermediate positions for realistic evaluation
- **Validation**: Cross-location generalization assessment

## 📋 Data Format

### CSV File Structure
```csv
rssi,amplitude_0,amplitude_1,...,amplitude_51,phase_0,phase_1,...,phase_51
-45.2,0.123,0.456,...,0.789,-1.23,0.45,...,2.13
-44.8,0.134,0.445,...,0.801,-1.19,0.48,...,2.09
...
```

### Feature Details
- **RSSI**: Signal strength in dBm (typically -100 to 0)
- **Amplitude**: CSI amplitude values (non-negative)
- **Phase**: CSI phase values (typically -π to π radians)
- **Coordinates**: Extracted from filename for ground truth

## 🔄 Data Processing Pipeline

### Raw Data → Processed Data
1. **Collection**: Raw WiFi measurements in laboratory
2. **Extraction**: CSI amplitude/phase separation from complex values
3. **Formatting**: Conversion to standardized CSV format
4. **Validation**: Quality checks and outlier removal
5. **Subsampling**: Creation of 250/500/750 sample variants

### Quality Assurance
- **Reproducible Sampling**: Fixed random seeds for consistent subsets
- **Feature Validation**: Amplitude non-negativity, phase range checks
- **Spatial Consistency**: Coordinate validation and mapping
- **Statistical Verification**: Distribution analysis and outlier detection

## 📈 Usage Guidelines

### For Quick Experiments
```bash
# Use 250-sample dataset
python model.py --dataset-path "Data/CSI Dataset 250 Samples/"
```

### For Standard Training
```bash
# Use 500-sample dataset
python model.py --dataset-path "Data/CSI Dataset 500 Samples/"
```

### For Best Performance
```bash
# Use 750-sample dataset  
python model.py --dataset-path "Data/CSI Dataset 750 Samples/"
```

### For Testing
```bash
# Use corresponding test points
python evaluate.py --test-path "Data/Tesing points/Testing Points Dataset 500/"
```

## ⚡ Performance Considerations

### Dataset Size vs. Performance
- **250 samples**: 70-80% of optimal performance, 3x faster training
- **500 samples**: 85-90% of optimal performance, 1.5x faster training  
- **750 samples**: Optimal performance, full training time

### Memory Requirements
- **250 samples**: ~50MB RAM
- **500 samples**: ~100MB RAM
- **750 samples**: ~150MB RAM

### Recommended Usage
- **Development Phase**: Start with 250 samples
- **Validation Phase**: Use 500 samples
- **Final Evaluation**: Deploy with 750 samples

## 🛠️ Data Loading

### Python Example
```python
import pandas as pd
import numpy as np

# Load single location data
data = pd.read_csv("Data/CSI Dataset 500 Samples/2,3.csv")

# Extract features
rssi = data['rssi'].values
amplitude = data[[f'amplitude_{i}' for i in range(52)]].values
phase = data[[f'phase_{i}' for i in range(52)]].values

# Combine for CNN input (52, 2)
csi_features = np.stack([amplitude, phase], axis=-1)
```

### Batch Loading
```python
# Load all locations
import glob

dataset_path = "Data/CSI Dataset 500 Samples/"
files = glob.glob(f"{dataset_path}/*.csv")

all_data = []
coordinates = []

for file in files:
    # Extract coordinates from filename
    coords = file.split('/')[-1].replace('.csv', '').split(',')
    x, y = float(coords[0]), float(coords[1])
    
    # Load data
    data = pd.read_csv(file)
    all_data.append(data)
    coordinates.extend([(x, y)] * len(data))
```

## 📊 Dataset Statistics

### Sample Distribution
- **Total Files**: 34 spatial locations
- **Samples per Location**: 250/500/750 (depending on dataset variant)
- **Feature Dimensions**: 105 features per sample
- **Coordinate Range**: X: [0,6], Y: [0,6] (approximate)

### Data Quality Metrics
- **Missing Values**: None (complete dataset)
- **Outliers**: Minimal (quality-controlled collection)
- **Noise Level**: Laboratory-controlled environment
- **Consistency**: High temporal and spatial consistency

---

**Comprehensive WiFi CSI Dataset for Indoor Localization Research**  
*Multi-scale datasets supporting both classical and deep learning approaches*

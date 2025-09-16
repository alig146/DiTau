# DiTau Analysis

A production-ready analysis framework for DiTau physics analysis, reorganized from the original `Ana_v7.ipynb` notebook.

## Project Structure

```
DiTau/
├── analysis/                    # Main analysis code
│   ├── analysis_utils.py       # Core analysis utilities and classes
│   ├── config.py               # Configuration management
│   ├── ml_utils.py             # Machine learning utilities
│   ├── plotting_utils.py       # Plotting and visualization
│   ├── run_analysis.py         # Main analysis script
│   └── Ana_v7.ipynb           # Original notebook (preserved)
├── utils/                      # General utilities
│   └── utils.py               # Existing utility functions
├── process_data/               # Data processing scripts
├── run/                       # Analysis runs and results
├── run2/                      # Additional analysis runs
├── requirements.txt           # Python dependencies
├── environment.yml           # Conda environment
└── README.md                 # This file
```

## Key Improvements

### 1. **Modular Design**
- **`analysis_utils.py`**: Core analysis classes and functions
  - `SampleDefinitions`: Sample IDs and configurations
  - `BranchDefinitions`: ROOT branch definitions
  - `DataLoader`: Data loading from ROOT files
  - `CutProcessor`: Analysis cuts
  - `VariableCalculator`: Physics variable calculations
  - `FakeFactorCalculator`: Fake factor calculations

- **`config.py`**: Centralized configuration management
  - `AnalysisConfig`: Main analysis parameters
  - `CutConfig`: Cut parameters
  - `MLConfig`: Machine learning parameters
  - `PlotConfig`: Plotting parameters

- **`ml_utils.py`**: Machine learning functionality
  - `MLTrainer`: XGBoost model training
  - `DataProcessor`: Data preparation for ML

- **`plotting_utils.py`**: Visualization tools
  - `Plotter`: Comprehensive plotting functions

### 2. **Production-Ready Features**
- **Logging**: Comprehensive logging throughout
- **Error Handling**: Robust error handling and recovery
- **Configuration**: Centralized, easily modifiable configuration
- **Documentation**: Extensive docstrings and comments
- **Type Hints**: Full type annotation for better code clarity

### 3. **Clean API**
- **`run_analysis.py`**: Main script with command-line interface
- **`DiTauAnalysis`**: Main analysis class orchestrating the pipeline
- **Modular functions**: Each component can be used independently

## Usage

### Basic Usage

```bash
# Run full analysis from scratch
python analysis/run_analysis.py

# Run analysis loading from cached data
python analysis/run_analysis.py --load-cache

# Run with debug logging
python analysis/run_analysis.py --log-level DEBUG
```

### Programmatic Usage

```python
from analysis.run_analysis import DiTauAnalysis
from analysis.config import get_config

# Initialize analysis
analysis = DiTauAnalysis()

# Run specific parts
analysis.load_weights()
analysis.load_mc_data()
analysis.load_data()
analysis.apply_cuts()
analysis.combine_years()
ml_results = analysis.run_ml_analysis()
```

### Configuration

Modify `analysis/config.py` to change:
- Data paths
- Analysis parameters
- Cut values
- ML hyperparameters
- Plot settings

## Installation

### Using Conda (Recommended)

```bash
# Create environment from existing file
conda env create -f environment.yml
conda activate ditau

# Install additional requirements
pip install -r requirements.txt
```

### Using pip

```bash
# Create virtual environment
python -m venv ditau_env
source ditau_env/bin/activate  # On Windows: ditau_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Key Features

### 1. **Data Loading**
- Automatic ROOT file discovery and loading
- Support for multiple datasets and years
- Efficient memory management
- Caching of raw data

### 2. **Analysis Pipeline**
- Configurable cuts and selections
- Physics variable calculations
- Fake factor application
- Cross-validation for ML

### 3. **Machine Learning**
- XGBoost integration
- SHAP value calculation
- Feature importance analysis
- ROC curve generation

### 4. **Visualization**
- Comprehensive plotting utilities
- Publication-ready figures
- Configurable plot styles
- Automatic plot saving

## Migration from Notebook

The original `Ana_v7.ipynb` notebook has been preserved and can still be used for interactive analysis. The new modular structure provides:

1. **Better organization**: Related functions grouped into logical modules
2. **Reusability**: Functions can be imported and used independently
3. **Maintainability**: Easier to modify and extend
4. **Testing**: Individual components can be tested separately
5. **Documentation**: Clear API with docstrings

## Development

### Adding New Features

1. **New analysis variables**: Add to `VariableCalculator` class
2. **New cuts**: Extend `CutProcessor` class
3. **New plots**: Add methods to `Plotter` class
4. **New ML models**: Extend `MLTrainer` class

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Use logging instead of print statements

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `numpy`, `pandas`: Data manipulation
- `uproot`, `awkward`: ROOT data handling
- `matplotlib`: Plotting
- `scikit-learn`, `xgboost`: Machine learning
- `shap`: Model interpretation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the coding style
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]
"""
DiTau Analysis Package

A production-ready analysis framework for DiTau physics analysis.
"""

__version__ = "1.0.0"
__author__ = "DiTau Analysis Team"

# Import main classes for easy access
from .config import get_config, get_cut_config, get_ml_config, get_plot_config
from .analysis_utils import (
    SampleDefinitions, BranchDefinitions, DataLoader, 
    CutProcessor, VariableCalculator, FakeFactorCalculator
)
from .ml_utils import MLTrainer, DataProcessor
from .plotting_utils import Plotter
from .run_analysis import DiTauAnalysis

__all__ = [
    'get_config', 'get_cut_config', 'get_ml_config', 'get_plot_config',
    'SampleDefinitions', 'BranchDefinitions', 'DataLoader', 
    'CutProcessor', 'VariableCalculator', 'FakeFactorCalculator',
    'MLTrainer', 'DataProcessor', 'Plotter', 'DiTauAnalysis'
]

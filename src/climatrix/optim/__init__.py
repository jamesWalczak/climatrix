"""Optimization module for hyperparameter tuning of reconstruction methods."""

try:
    from .bayesian import HParamFinder as HParamFinder
except ImportError:
    pass
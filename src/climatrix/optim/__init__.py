"""Optimization module for hyperparameter tuning."""

try:
    from .bayesian import HParamFinder as HParamFinder
except ImportError:
    pass

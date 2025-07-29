"""
Hyperparameter type system for reconstruction methods.
"""
from __future__ import annotations

from typing import Generic, TypeVar, Any, Union


T = TypeVar('T')


class Hyperparameter(Generic[T]):
    """
    Type annotation for hyperparameters in reconstruction classes.
    
    Examples
    --------
    Define hyperparameters as class attributes with bounds:
    
    >>> class SomeRec(BaseReconstructor):
    ...     # Type annotations for IDE support
    ...     k: Hyperparameter[int]
    ...     power: Hyperparameter[float]
    ...     
    ...     # Hyperparameter specifications
    ...     _hparam_k = {'type': int, 'bounds': (1, 10)}
    ...     _hparam_power = {'type': float, 'bounds': (0.5, 5.0)}
    """
    
    def __init__(self, param_type: type, bounds: tuple = None, values: list = None):
        self.param_type = param_type
        self.bounds = bounds  
        self.values = values
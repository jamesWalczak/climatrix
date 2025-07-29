"""
Hyperparameter type system for reconstruction methods.
"""
from __future__ import annotations

from typing import Generic, TypeVar, Any, Union, get_type_hints, get_origin, get_args
try:
    from typing_extensions import Annotated, get_args as get_args_ext
except ImportError:
    from typing import Annotated, get_args as get_args_ext


T = TypeVar('T')


class Hyperparameter(Generic[T]):
    """
    Type annotation for hyperparameters in reconstruction classes using Annotated.
    
    Examples
    --------
    Define hyperparameters with annotations that include bounds or values:
    
    >>> from typing_extensions import Annotated
    >>> 
    >>> class SomeRec(BaseReconstructor):
    ...     k: Annotated[Hyperparameter[int], {'type': int, 'bounds': (1, 10)}]
    ...     power: Annotated[Hyperparameter[float], {'type': float, 'bounds': (0.5, 5.0)}]
    ...     mode: Annotated[Hyperparameter[str], {'type': str, 'values': ['fast', 'slow']}]
    """
    
    def __init__(self, param_type: type, bounds: tuple = None, values: list = None):
        self.param_type = param_type
        self.bounds = bounds
        self.values = values
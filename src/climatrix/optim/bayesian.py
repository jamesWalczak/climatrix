"""Bayesian optimization for hyperparameter tuning of reconstruction methods."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Collection, Union

import numpy as np

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.decorators.runtime import raise_if_not_installed
from climatrix.comparison import Comparison

log = logging.getLogger(__name__)

# Module-level constants
DEFAULT_BAD_SCORE = -1e6


class MetricType(StrEnum):
    """Supported metrics for hyperparameter optimization."""
    MAE = "mae"
    MSE = "mse" 
    RMSE = "rmse"


def get_hparams_bounds(method: str) -> dict[str, tuple[float, float]]:
    """
    Get default hyperparameter bounds for a given reconstruction method.
    
    Parameters
    ----------
    method : str
        The reconstruction method name (e.g., 'idw', 'ok', 'sinet', 'siren').
        
    Returns
    -------
    dict[str, tuple[float, float]]
        Dictionary mapping parameter names to (min, max) bounds.
        
    Raises
    ------
    ValueError
        If the method is not supported.
    """
    from climatrix.reconstruct.base import BaseReconstructor
    
    reconstruction_class = BaseReconstructor.get(method)
    # Access hparams directly from the class since they are class-level definitions
    hparams = reconstruction_class.hparams()
    
    bounds = {}
    for param_name, param_def in hparams.items():
        if 'bounds' in param_def:
            bounds[param_name] = param_def['bounds']
        elif 'values' in param_def:
            # Handle categorical parameters - for now skip them
            continue
    
    return bounds


class HParamFinder:
    """
    Bayesian hyperparameter optimization for reconstruction methods.
    
    This class uses Bayesian optimization to find optimal hyperparameters
    for various reconstruction methods.
    
    Parameters
    ----------
    train_dset : BaseClimatrixDataset
        Training dataset used for optimization.
    val_dset : BaseClimatrixDataset
        Validation dataset used for optimization.
    metric : str, optional
        Evaluation metric to optimize. Default is "mae".
        Supported metrics: "mae", "mse", "rmse".
    method : str, optional
        Reconstruction method to optimize. Default is "idw".
    exclude : str or Collection[str], optional
        Parameter(s) to exclude from optimization.
    include : str or Collection[str], optional
        Parameter(s) to include in optimization. If specified, only these
        parameters will be optimized.
    explore : float, optional
        Exploration vs exploitation trade-off parameter in (0, 1).
        Higher values favor exploration. Default is 0.9.
    n_iters : int, optional
        Total number of optimization iterations. Default is 100.
    bounds : dict, optional
        Custom parameter bounds. Overrides default bounds for the method.
    random_seed : int, optional
        Random seed for reproducible optimization. Default is 42.
        
    Attributes
    ----------
    train_dset : BaseClimatrixDataset
        Training dataset.
    val_dset : BaseClimatrixDataset  
        Validation dataset.
    metric : MetricType
        Evaluation metric.
    method : str
        Reconstruction method.
    bounds : dict
        Parameter bounds for optimization.
    n_init_points : int
        Number of initial random points.
    n_iter : int
        Number of Bayesian optimization iterations.
    random_seed : int
        Random seed for optimization.
    """
    
    def __init__(
        self,
        train_dset: BaseClimatrixDataset,
        val_dset: BaseClimatrixDataset,
        *,
        metric: str = "mae", 
        method: str = "idw",
        exclude: Union[str, Collection[str], None] = None,
        include: Union[str, Collection[str], None] = None,
        explore: float = 0.9,
        n_iters: int = 100,
        bounds: dict[str, tuple[float, float]] | None = None,
        random_seed: int = 42,
    ):
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.metric = MetricType(metric.lower().strip())
        self.method = method.lower().strip()
        self.random_seed = random_seed
        
        # Validate inputs
        self._validate_inputs(explore, n_iters)
        
        # Get default bounds and apply customizations
        default_bounds = get_hparams_bounds(self.method) if bounds is None else {}
        self.bounds = {**default_bounds, **(bounds or {})}
        self._filter_parameters(include, exclude)
        
        # Compute init points and iterations based on explore parameter
        self.n_init_points = max(1, int(n_iters * explore))
        self.n_iter = n_iters - self.n_init_points
        
        log.debug(
            "HParamFinder initialized: method=%s, metric=%s, "
            "n_init_points=%d, n_iter=%d, bounds=%s",
            self.method, self.metric, self.n_init_points, self.n_iter, self.bounds
        )
    
    def _validate_inputs(self, explore: float, n_iters: int) -> None:
        """Validate input parameters."""
        if not isinstance(self.train_dset, BaseClimatrixDataset):
            raise TypeError("train_dset must be a BaseClimatrixDataset")
        if not isinstance(self.val_dset, BaseClimatrixDataset):
            raise TypeError("val_dset must be a BaseClimatrixDataset") 
        if not 0 < explore < 1:
            raise ValueError("explore must be in the range (0, 1)")
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")
    
    def _filter_parameters(
        self, 
        include: Union[str, Collection[str], None],
        exclude: Union[str, Collection[str], None]
    ) -> None:
        """Filter parameters based on include/exclude lists."""
        if include is not None and exclude is not None:
            include_set = {include} if isinstance(include, str) else set(include)
            exclude_set = {exclude} if isinstance(exclude, str) else set(exclude)
            common_keys = include_set.intersection(exclude_set)
            if common_keys:
                raise ValueError(f"Cannot specify same parameters in both include and exclude: {common_keys}")
        
        if include is not None:
            if isinstance(include, str):
                include = [include]
            # Keep only included parameters
            filtered_bounds = {}
            for param in include:
                if param in self.bounds:
                    filtered_bounds[param] = self.bounds[param]
                else:
                    log.warning("Parameter '%s' not found in bounds for method '%s'", param, self.method)
            self.bounds = filtered_bounds
            
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            # Remove excluded parameters
            for param in exclude:
                if param in self.bounds:
                    del self.bounds[param]
                else:
                    log.warning("Parameter '%s' not found in bounds for method '%s'", param, self.method)
    
    def _evaluate_params(self, **params) -> float:
        """
        Evaluate a set of hyperparameters.
        
        Parameters
        ----------
        **params
            Hyperparameters to evaluate.
            
        Returns
        -------
        float
            Negative metric value (since BayesianOptimization maximizes).
        """
        try:
            # Get parameter types from the reconstruction class
            from climatrix.reconstruct.base import BaseReconstructor
            reconstruction_class = BaseReconstructor.get(self.method)
            instance = reconstruction_class.__new__(reconstruction_class)
            hparams_def = instance.hparams
            
            # Convert parameters to appropriate types
            processed_params = {}
            for key, value in params.items():
                if key in hparams_def:
                    param_type = hparams_def[key]['type']
                    if param_type == int:
                        processed_params[key] = int(round(value))
                    elif param_type == bool:
                        processed_params[key] = bool(round(value))
                    elif param_type == float:
                        processed_params[key] = value
                    else:
                        processed_params[key] = value
                else:
                    processed_params[key] = value
            
            log.debug("Evaluating parameters: %s", processed_params)
            
            # Perform reconstruction with given parameters
            reconstructed = self.train_dset.reconstruct(
                target=self.val_dset.domain,
                method=self.method,
                **processed_params
            )
            
            # Compute metric using Comparison class
            comparison = Comparison(reconstructed, self.val_dset)
            score = comparison.compute(self.metric.value)
            
            log.debug("Score for params %s: %f", processed_params, score)
            
            # Return negative score for maximization
            return -score
            
        except Exception as e:
            log.warning("Error evaluating parameters %s: %s", params, e)
            # Return a very bad score
            return DEFAULT_BAD_SCORE
    
    @raise_if_not_installed("bayesian-optimization")
    def optimize(self) -> dict[str, Any]:
        """
        Run Bayesian optimization to find optimal hyperparameters.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'best_params': Best hyperparameters found (with correct types)
            - 'best_score': Best score achieved (negative metric value)  
            - 'history': Optimization history
            - 'metric_name': Name of the optimized metric
            - 'method': Reconstruction method used
        """
        from bayes_opt import BayesianOptimization
        
        log.info("Starting Bayesian optimization for method '%s'", self.method)
        log.info("Bounds: %s", self.bounds)
        log.info("Using %d initial points and %d iterations", self.n_init_points, self.n_iter)
        
        # Handle categorical parameters (like variogram_model for kriging)
        categorical_params = {}
        numeric_bounds = {}
        
        for param, bound in self.bounds.items():
            if isinstance(bound, dict) and "values" in bound:
                # Categorical parameter
                categorical_params[param] = bound["values"]
            else:
                # Numeric parameter
                numeric_bounds[param] = bound
        
        if categorical_params:
            log.warning("Categorical parameters not yet fully supported: %s", categorical_params)
            # For now, just optimize numeric parameters
        
        if not numeric_bounds:
            raise ValueError("No numeric parameters to optimize")
        
        # Create and run optimizer
        optimizer = BayesianOptimization(
            f=self._evaluate_params,
            pbounds=numeric_bounds,
            random_state=self.random_seed,
        )
        
        optimizer.maximize(
            init_points=self.n_init_points,
            n_iter=self.n_iter,
        )
        
        # Get best parameters and convert to appropriate types
        best_params_raw = optimizer.max['params']
        from climatrix.reconstruct.base import BaseReconstructor
        reconstruction_class = BaseReconstructor.get(self.method)
        instance = reconstruction_class.__new__(reconstruction_class)
        hparams_def = instance.hparams
        
        best_params = {}
        for key, value in best_params_raw.items():
            if key in hparams_def:
                param_type = hparams_def[key]['type']
                if param_type == int:
                    best_params[key] = int(round(value))
                elif param_type == bool:
                    best_params[key] = bool(round(value))
                elif param_type == float:
                    best_params[key] = value
                else:
                    best_params[key] = value
            else:
                best_params[key] = value
        
        best_score = optimizer.max['target']
        
        log.info("Optimization completed. Best score: %f", best_score)
        log.info("Best parameters: %s", best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'metric_name': self.metric.value,
            'method': self.method,
            'history': [{'params': res['params'], 'target': res['target']} 
                       for res in optimizer.res],
        }
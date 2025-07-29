"""Bayesian optimization for hyperparameter tuning of reconstruction methods."""

from __future__ import annotations

import logging
from typing import Any, Collection, Union

import numpy as np

# Import BaseClimatrixDataset directly to avoid full climatrix import issues
from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.decorators.runtime import raise_if_not_installed

log = logging.getLogger(__name__)


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
    bounds = {
        "idw": {
            "power": (0.5, 5.0),
            "k": (1, 20),
            "k_min": (1, 10),
        },
        "ok": {
            # Common pykrige parameters
            "variogram_model": {
                # This is categorical, will be handled differently
                "values": ["linear", "power", "gaussian", "spherical", "exponential"]
            },
            "nlags": (4, 20),
            "weight": (0.0, 1.0),
        },
        "sinet": {
            "lr": (1e-5, 1e-2),
            "batch_size": (64, 1024),
            "num_epochs": (1000, 10000),
            "mse_loss_weight": (1e1, 1e4),
            "eikonal_loss_weight": (1e0, 1e3),
            "laplace_loss_weight": (1e1, 1e3),
        },
        "siren": {
            "lr": (1e-5, 1e-2), 
            "batch_size": (64, 1024),
            "num_epochs": (1000, 10000),
            "hidden_dim": (128, 512),
            "num_layers": (3, 8),
        }
    }
    
    method = method.lower()
    if method not in bounds:
        raise ValueError(
            f"Unknown reconstruction method: {method}. "
            f"Supported methods are: {list(bounds.keys())}"
        )
    
    return bounds[method]


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
        
    Attributes
    ----------
    train_dset : BaseClimatrixDataset
        Training dataset.
    val_dset : BaseClimatrixDataset  
        Validation dataset.
    metric : str
        Evaluation metric.
    method : str
        Reconstruction method.
    bounds : dict
        Parameter bounds for optimization.
    n_init_points : int
        Number of initial random points.
    n_iter : int
        Number of Bayesian optimization iterations.
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
    ):
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.metric = metric.lower()
        self.method = method.lower()
        
        # Validate inputs
        self._validate_inputs(explore, n_iters)
        
        # Get default bounds and apply customizations
        self.bounds = bounds or get_hparams_bounds(self.method)
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
        if self.metric not in ["mae", "mse", "rmse"]:
            raise ValueError(f"Unsupported metric: {self.metric}")
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
            raise ValueError("Cannot specify both include and exclude parameters")
        
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
            
        elif exclude is not None:
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
            # Perform reconstruction with given parameters
            reconstructed = self.train_dset.reconstruct(
                target=self.val_dset.domain,
                method=self.method,
                **params
            )
            
            # Compute metric
            diff = reconstructed.da.values - self.val_dset.da.values
            
            if self.metric == "mae":
                score = np.nanmean(np.abs(diff))
            elif self.metric == "mse":
                score = np.nanmean(diff ** 2)
            elif self.metric == "rmse":
                score = np.sqrt(np.nanmean(diff ** 2))
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            # Return negative score for maximization
            return -score
            
        except Exception as e:
            log.warning("Error evaluating parameters %s: %s", params, e)
            # Return a very bad score
            return -1e6
    
    @raise_if_not_installed("bayesian-optimization")
    def optimize(self) -> dict[str, Any]:
        """
        Run Bayesian optimization to find optimal hyperparameters.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'best_params': Best hyperparameters found
            - 'best_score': Best score achieved (negative metric value)
            - 'history': Optimization history
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
            random_state=42,
        )
        
        optimizer.maximize(
            init_points=self.n_init_points,
            n_iter=self.n_iter,
        )
        
        best_params = optimizer.max['params']
        best_score = optimizer.max['target']
        
        log.info("Optimization completed. Best score: %f", best_score)
        log.info("Best parameters: %s", best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': [{'params': res['params'], 'target': res['target']} 
                       for res in optimizer.res],
        }
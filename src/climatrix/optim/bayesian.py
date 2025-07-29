"""Bayesian optimization for hyperparameter tuning of reconstruction methods."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from climatrix.decorators import raise_if_not_installed

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset

log = logging.getLogger(__name__)


class HParamFinder:
    """
    Hyperparameter optimization using Bayesian optimization for reconstruction methods.
    
    Parameters
    ----------
    train_dset : BaseClimatrixDataset
        Training dataset for optimization.
    val_dset : BaseClimatrixDataset  
        Validation dataset for optimization.
    metric : str, optional
        Evaluation metric to optimize. Default is "mae".
    method : str, optional
        Reconstruction method name. Default is "idw".
    exclude : str or list of str, optional
        Hyperparameters to exclude from optimization.
    include : str or list of str, optional
        Hyperparameters to include in optimization. 
        If provided, only these will be optimized.
    explore : float, optional
        Exploration vs exploitation trade-off (0 < explore < 1). Default is 0.9.
    n_iters : int, optional
        Total number of optimization iterations. Default is 100.
    **bounds_kwargs
        Custom bounds for hyperparameters to override defaults.
        
    Attributes
    ----------
    train_dset : BaseClimatrixDataset
        Training dataset.
    val_dset : BaseClimatrixDataset
        Validation dataset.
    metric : str
        Evaluation metric.
    method : str
        Reconstruction method name.
    explore : float
        Exploration parameter.
    n_iters : int
        Total iterations.
    init_points : int
        Number of random initialization points.
    n_iter : int
        Number of Bayesian optimization iterations.
    bounds : dict
        Hyperparameter bounds for optimization.
    """
    
    def __init__(
        self,
        train_dset: BaseClimatrixDataset,
        val_dset: BaseClimatrixDataset,
        metric: str = "mae",
        method: str = "idw", 
        exclude: Optional[Union[str, List[str]]] = None,
        include: Optional[Union[str, List[str]]] = None,
        explore: float = 0.9,
        n_iters: int = 100,
        **bounds_kwargs
    ):
        if not (0 < explore < 1):
            raise ValueError("explore must be between 0 and 1")
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")
            
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.metric = metric
        self.method = method
        self.explore = explore
        self.n_iters = n_iters
        
        # Calculate init_points and n_iter based on explore and n_iters
        self.init_points = max(1, int(n_iters * explore))
        self.n_iter = n_iters - self.init_points
        
        log.debug(f"Using {self.init_points} init_points and {self.n_iter} n_iter")
        
        # Get default bounds for the method
        self.bounds = self._get_method_bounds(method, bounds_kwargs)
        
        # Filter bounds based on include/exclude
        self.bounds = self._filter_bounds(self.bounds, include, exclude)
        
        if not self.bounds:
            raise ValueError("No hyperparameters to optimize after filtering")
            
        log.info(f"Optimizing hyperparameters: {list(self.bounds.keys())}")
        
    def _get_method_bounds(self, method: str, bounds_kwargs: Dict[str, Any]) -> Dict[str, tuple]:
        """Get default bounds for the specified method."""
        # Get reconstructor class and its default bounds
        reconstructor_class = self._get_reconstructor_class(method)
        default_bounds = reconstructor_class.get_hparams_bounds()
        
        # Override with any custom bounds
        for param, bounds in bounds_kwargs.items():
            if param in default_bounds or param not in default_bounds:
                # Allow custom bounds even for params not in defaults
                default_bounds[param] = bounds
                log.debug(f"Override bounds for {param}: {bounds}")
                
        return default_bounds
        
    def _filter_bounds(
        self,
        bounds: Dict[str, tuple],
        include: Optional[Union[str, List[str]]],
        exclude: Optional[Union[str, List[str]]]
    ) -> Dict[str, tuple]:
        """Filter bounds based on include/exclude parameters."""
        if include is not None:
            if isinstance(include, str):
                include = [include]
            bounds = {k: v for k, v in bounds.items() if k in include}
            
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            bounds = {k: v for k, v in bounds.items() if k not in exclude}
            
        return bounds
        
    @raise_if_not_installed("bayes_opt")
    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find best hyperparameters.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'best_params': Best hyperparameters found
            - 'best_score': Best validation score achieved
            - 'optimizer': The BayesianOptimization object for further analysis
        """
        from bayes_opt import BayesianOptimization
        
        def objective(**params):
            """Objective function to minimize/maximize."""
            return self._evaluate_params(params)
            
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=self.bounds,
            random_state=42,
            verbose=2
        )
        
        log.info(f"Starting Bayesian optimization with {self.init_points} random points "
                f"and {self.n_iter} optimization steps")
        
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter
        )
        
        best_params = optimizer.max['params']
        best_score = optimizer.max['target']
        
        log.info(f"Optimization completed. Best score: {best_score:.6f}")
        log.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimizer': optimizer
        }
        
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """
        Evaluate hyperparameters by training and validating a reconstruction method.
        
        Parameters
        ---------- 
        params : dict
            Hyperparameters to evaluate.
            
        Returns
        -------
        float
            Validation score (higher is better for the optimizer).
        """
        try:
            # Convert float parameters to integers where needed
            processed_params = self._process_params(params)
            
            # Get reconstructor class
            reconstructor_class = self._get_reconstructor_class(self.method)
            
            # Create reconstructor with training data and hyperparameters
            reconstructor = reconstructor_class(
                self.train_dset, 
                self.val_dset.domain,
                **processed_params
            )
            
            # Perform reconstruction
            reconstructed = reconstructor.reconstruct()
            
            # Calculate validation metric
            score = self._calculate_metric(reconstructed, self.val_dset, self.metric)
            
            # For optimization, we want to maximize, but some metrics like MAE should be minimized
            if self.metric in ["mae", "mse", "rmse"]:
                score = -score  # Convert to maximization problem
                
            log.debug(f"Evaluated params {processed_params}: score = {score:.6f}")
            
            return score
            
        except Exception as e:
            log.warning(f"Evaluation failed for params {params}: {e}")
            # Return a very bad score for failed evaluations
            return -1e10
            
    def _process_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process parameters to convert types as needed."""
        processed = {}
        
        for key, value in params.items():
            # Convert parameters that should be integers
            if key in ['k', 'k_min']:
                processed[key] = int(round(value))
            else:
                processed[key] = value
                
        return processed
        
    def _get_reconstructor_class(self, method: str):
        """Get the reconstructor class for the given method."""
        if method == "idw":
            from climatrix.reconstruct.idw import IDWReconstructor
            return IDWReconstructor
        elif method == "kriging":
            from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor
            return OrdinaryKrigingReconstructor
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
            
    def _calculate_metric(
        self, 
        reconstructed: BaseClimatrixDataset, 
        validation: BaseClimatrixDataset,
        metric: str
    ) -> float:
        """Calculate validation metric between reconstructed and validation data."""
        # Get overlapping points between reconstructed and validation data
        reconstructed_values = reconstructed.da.values.flatten()
        validation_values = validation.da.values.flatten()
        
        # Handle different array sizes by taking minimum length
        min_len = min(len(reconstructed_values), len(validation_values))
        reconstructed_values = reconstructed_values[:min_len]
        validation_values = validation_values[:min_len]
        
        # Handle NaN values
        valid_mask = ~(np.isnan(reconstructed_values) | np.isnan(validation_values))
        
        if not np.any(valid_mask):
            log.warning("No valid overlapping points for metric calculation")
            return np.inf if metric in ["mae", "mse", "rmse"] else 0.0
            
        recon_valid = reconstructed_values[valid_mask]
        val_valid = validation_values[valid_mask]
        
        if metric == "mae":
            return np.mean(np.abs(recon_valid - val_valid))
        elif metric == "mse":
            return np.mean((recon_valid - val_valid) ** 2)
        elif metric == "rmse":
            return np.sqrt(np.mean((recon_valid - val_valid) ** 2))
        elif metric == "r2":
            ss_res = np.sum((val_valid - recon_valid) ** 2)
            ss_tot = np.sum((val_valid - np.mean(val_valid)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")


def get_hparams_bounds(method: str) -> Dict[str, tuple]:
    """
    Get default hyperparameter bounds for reconstruction methods.
    
    This is a convenience function that delegates to the appropriate
    reconstructor class's get_hparams_bounds() method.
    
    Parameters
    ----------
    method : str
        Name of the reconstruction method ("idw", "kriging", etc.).
        
    Returns
    -------
    Dict[str, tuple]
        Dictionary mapping hyperparameter names to (min, max) bounds.
        
    Examples
    --------
    >>> bounds = get_hparams_bounds("idw")
    >>> print(bounds)
    {'power': (0.5, 5.0), 'k': (3, 20), 'k_min': (1, 10)}
    """
    if method == "idw":
        from climatrix.reconstruct.idw import IDWReconstructor
        return IDWReconstructor.get_hparams_bounds()
    elif method == "kriging":
        from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor
        return OrdinaryKrigingReconstructor.get_hparams_bounds()
    else:
        raise ValueError(f"Unknown reconstruction method: {method}. "
                        f"Supported methods: idw, kriging")
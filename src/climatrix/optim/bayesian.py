"""Bayesian optimization for hyperparameter tuning."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Collection, Union

import numpy as np

from climatrix.comparison import Comparison
from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.decorators.runtime import raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor

log = logging.getLogger(__name__)

# Module-level constants
DEFAULT_BAD_SCORE = -1e6


class MetricType(StrEnum):
    """Supported metrics for hyperparameter optimization."""

    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"


class HParamFinder:
    """
    Bayesian hyperparameter optimization for reconstruction methods.

    This class uses Bayesian optimization to find optimal hyperparameters
    for various reconstruction methods.

    Parameters
    ----------
    method : str
        Reconstruction method to optimize.
    train_dset : BaseClimatrixDataset
        Training dataset used for optimization.
    val_dset : BaseClimatrixDataset
        Validation dataset used for optimization.
    metric : str, optional
        Evaluation metric to optimize. Default is "mae".
        Supported metrics: "mae", "mse", "rmse".
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
        method: str,
        train_dset: BaseClimatrixDataset,
        val_dset: BaseClimatrixDataset,
        *,
        metric: str = "mae",
        exclude: str | Collection[str] | None = None,
        include: str | Collection[str] | None = None,
        explore: float = 0.9,
        n_iters: int = 100,
        bounds: dict[str, tuple[float, float]] | None = None,
        random_seed: int = 42,
        verbose: int = 0,
    ):
        self.mapping: dict[str, dict[int, str]] = {}
        self.result: dict[str, Any] = {}
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.metric = MetricType(metric.lower().strip())
        self.method = method.lower().strip()
        self.method_hparams = BaseReconstructor.get(self.method).get_hparams()
        self.random_seed = random_seed

        self._validate_inputs(explore, n_iters)

        self._compute_bounds(bounds)
        self._filter_parameters(include, exclude)

        self.n_init_points = max(1, int(n_iters * explore))
        self.n_iter = n_iters - self.n_init_points

        self.verbose = verbose

        log.debug(
            "HParamFinder initialized: method=%s, metric=%s, "
            "n_init_points=%d, n_iter=%d, bounds=%s",
            self.method,
            self.metric,
            self.n_init_points,
            self.n_iter,
            self.bounds,
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

    def _compute_bounds(self, user_defined_bounds: dict) -> None:
        """
        Compute parameter bounds for optimization.

        Notes
        -----
        - Handles properly categorical parameters.
        - Uses default bounds as defined for `Hyperparameter`
        if not provided.
        """
        from climatrix.reconstruct.base import BaseReconstructor

        user_defined_bounds = user_defined_bounds or {}
        method = BaseReconstructor.get(self.method)
        hparam_defs: dict = method.get_hparams()
        bounds = {}
        for param_name, param_def in hparam_defs.items():
            if "bounds" in param_def:
                bounds[param_name] = tuple(param_def["bounds"])
                method.update_bounds(bounds={param_name: param_def["bounds"]})
            elif "values" in param_def:
                self.mapping[param_name] = {
                    i: v for i, v in enumerate(param_def["values"])
                }
                bounds[param_name] = ("0", str(len(param_def["values"]) - 1))
                method.update_bounds(values={param_name: param_def["values"]})
        # NOTE: user-defined bounds override defaults
        for param_name, param_value in user_defined_bounds.items():
            if isinstance(param_value, tuple):
                bounds[param_name] = tuple(param_value)
            elif isinstance(param_value, list):
                self.mapping[param_name] = {
                    i: v for i, v in enumerate(param_value)
                }
                bounds[param_name] = ("0", str(len(param_value) - 1))
            else:
                raise TypeError(
                    f"Invalid bounds for parameter '{param_name}': {param_value}"
                )

        if not bounds:
            raise ValueError(f"No bounds defined for method '{self.method}'")
        self.bounds = bounds

    def _filter_parameters(
        self,
        include: str | Collection[str] | None,
        exclude: str | Collection[str] | None,
    ) -> None:
        """Filter parameters based on include/exclude lists."""
        if include is not None and exclude is not None:
            include_set = (
                {include} if isinstance(include, str) else set(include)
            )
            exclude_set = (
                {exclude} if isinstance(exclude, str) else set(exclude)
            )
            common_keys = include_set.intersection(exclude_set)
            if common_keys:
                raise ValueError(
                    f"Cannot specify same parameters in both include and exclude: {common_keys}"
                )

        if include is not None:
            if isinstance(include, str):
                include = [include]
            filtered_bounds = {}
            for param in include:
                if param in self.bounds:
                    filtered_bounds[param] = self.bounds[param]
                else:
                    log.warning(
                        "Parameter '%s' not found in bounds for method '%s'",
                        param,
                        self.method,
                    )
            self.bounds = filtered_bounds

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for param in exclude:
                if param in self.bounds:
                    del self.bounds[param]
                else:
                    log.warning(
                        "Parameter '%s' not found in bounds for method '%s'",
                        param,
                        self.method,
                    )

    def _map_bo_output_to_valid_params(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        valid_params = {}
        for param_name, param_value in params.items():
            if param_name in self.mapping:
                valid_params[param_name] = self.mapping[param_name][
                    int(param_value)
                ]
            else:
                hparam = self.method_hparams.get(param_name)
                if hparam is not None:
                    valid_params[param_name] = hparam["type"](param_value)

        return valid_params

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
        params = self._map_bo_output_to_valid_params(params)
        try:
            log.debug("Evaluating parameters: %s", params)
            reconstructed = self.train_dset.reconstruct(
                target=self.val_dset.domain, method=self.method, **params
            )

            comparison = Comparison(reconstructed, self.val_dset)
            score = comparison.compute(self.metric.value)

            log.debug("Score for params %s: %f", params, score)
            # NOTE: Return negative score for maximization
            return -score

        except Exception as e:
            log.warning("Error evaluating parameters %s: %s", params, e)
            return DEFAULT_BAD_SCORE

    @raise_if_not_installed("bayes_opt")
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
        log.info(
            "Using %d initial points and %d iterations",
            self.n_init_points,
            self.n_iter,
        )
        optimizer = BayesianOptimization(
            f=self._evaluate_params,
            pbounds=self.bounds,
            random_state=self.random_seed,
            verbose=self.verbose,
        )

        optimizer.maximize(
            init_points=self.n_init_points,
            n_iter=self.n_iter,
        )
        optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer.space)

        best_params = optimizer.max["params"]
        best_params = self._map_bo_output_to_valid_params(best_params)
        best_score = optimizer.max["target"]

        log.info("Optimization completed. Best score: %f", best_score)
        log.info("Best parameters: %s", best_params)

        self.result = {
            "best_params": best_params,
            "best_score": best_score,
            "metric_name": self.metric.value,
            "method": self.method,
            "history": [
                {"params": res["params"], "target": res["target"]}
                for res in optimizer.res
            ],
        }
        return self.result

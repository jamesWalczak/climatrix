"""Bayesian optimization for hyperparameter tuning."""

from __future__ import annotations

import logging
from collections import OrderedDict
from enum import StrEnum
from numbers import Number
from typing import Any, Callable, Collection

import numpy as np

from climatrix.comparison import Comparison
from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.decorators.runtime import raise_if_not_installed
from climatrix.optim.hyperparameter import Hyperparameter
from climatrix.reconstruct.base import BaseReconstructor

log = logging.getLogger(__name__)

# Module-level constants
DEFAULT_BAD_SCORE = 1e6


class MetricType(StrEnum):
    """Supported metrics for hyperparameter optimization."""

    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"


def update_bounds(
    method: BaseReconstructor,
    param_name: str,
    lower: Number | None,
    upper: Number | None,
    values: list | None = None,
):
    if all([lower is None, upper is None, values is None]):
        raise ValueError(
            "At least one of lower, upper, or values must be provided"
        )
    if values is not None and (lower is not None or upper is not None):
        raise ValueError("Cannot specify both bounds and values")

    if not isinstance(hparam := getattr(method, param_name), Hyperparameter):
        return

    if values is not None:
        hparam.values = values
    else:
        hparam.bounds = (lower, upper)


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
    n_iters : int, optional
        Total number of optimization iterations. Default is 100.
    bounds : dict, optional
        Custom parameter bounds. Overrides default bounds for the method.
    random_seed : int, optional
        Random seed for reproducible optimization. Default is 42.
    reconstructor_kwargs : dict, optional
        Additional keyword arguments for the reconstructor.
        The optimizable parameters will be overridden by the optimizer.

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
    n_iter : int
        Number of optimization iterations.
    random_seed : int
        Random seed for optimization.
    verbose : int
        Verbosity level for logging (0 - silent, 1 - info, 2 - debug).
    n_startup_trials : int
        Number of startup trials for the optimizer.
    n_warmup_steps : int
        Number of warmup steps before starting optimization.
    reconstructor_kwargs : dict
        Additional keyword arguments for the reconstructor.
    result : dict
        Dictionary containing optimization results:
        - 'best_params': Best hyperparameters found (with correct types)
        - 'best_score': Best score achieved (negative metric value)
        - 'metric_name': Name of the optimized metric
        - 'method': Reconstruction method used
        - 'n_trials': Total number of trials performed
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
        bounds: dict[str, tuple[float, float]] | None = None,
        random_seed: int = 42,
        n_iters: int = 100,
        verbose: int = 0,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 10,
        scoring_callback: (
            Callable[[int, dict[str, Any], float], float] | None
        ) = None,
        reconstructor_kwargs: dict[str, Any] | None = None,
    ):
        self.mapping: dict[str, dict[int, str]] = {}
        self.result: dict[str, Any] = {}
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.metric = MetricType(metric.lower().strip())
        self.method = method.lower().strip()
        self.method_hparams = BaseReconstructor.get(self.method).get_hparams()
        self.random_seed = random_seed

        self._validate_inputs(n_iters)

        self._compute_bounds(bounds)
        self._filter_parameters(include, exclude)

        self.n_iters = n_iters

        self.verbose = verbose
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.scoring_callback = scoring_callback or (
            lambda trial_num, params, score: score
        )
        self.reconstructor_kwargs = reconstructor_kwargs or {}

        log.debug(
            "HParamFinder initialized: method=%s, metric=%s, "
            "n_iter=%d, bounds=%s",
            self.method,
            self.metric,
            self.n_iters,
            self.bounds,
        )

    def _validate_inputs(self, n_iters: int) -> None:
        """Validate input parameters."""
        if not isinstance(self.train_dset, BaseClimatrixDataset):
            raise TypeError("train_dset must be a BaseClimatrixDataset")
        if not isinstance(self.val_dset, BaseClimatrixDataset):
            raise TypeError("val_dset must be a BaseClimatrixDataset")
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")

    def _suggest_arguments(self, trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for the current trial.

        Parameters
        ----------
        trial : optuna.Trial
            Current optimization trial.

        Returns
        -------
        dict[str, Any]
            Suggested hyperparameters with correct types.
        """
        suggested_params = {}
        for param_name, param_def in self.bounds.items():
            # NOTE: if list - suggest categorical parameter
            if isinstance(param_def, list):
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, param_def
                )
            else:
                # Otherwise suggest numeric parameter with bounds
                if not isinstance(param_def, tuple):
                    raise TypeError(
                        f"Invalid bounds for parameter '{param_name}': {param_def}"
                    )
                low, high, dtype = param_def
                if issubclass(dtype, int):
                    if low is None:
                        # NOTE: Bounds range cannot exceed int limits
                        low = np.iinfo(dtype).min // 2 + 1
                    if high is None:
                        high = np.iinfo(dtype).max // 2 - 1
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, low, high
                    )
                elif issubclass(dtype, float):
                    if low is None:
                        # NOTE: Bounds range cannot exceed float limits
                        low = np.finfo(dtype).min / 2 + 1
                    if high is None:
                        high = np.finfo(dtype).max / 2 - 1
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, low, high
                    )
                else:
                    raise TypeError(
                        f"Unsupported parameter type for '{param_name}': {dtype}"
                    )
        log.info(
            "Suggested parameters for trial %s: %s",
            trial.number,
            suggested_params,
        )
        return suggested_params

    @raise_if_not_installed("optuna")
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
        bounds = OrderedDict()
        for param_name, param_def in hparam_defs.items():
            if "bounds" in param_def:
                if issubclass(param_def["type"], int):
                    bounds[param_name] = (
                        param_def["bounds"][0],
                        param_def["bounds"][1],
                        int,
                    )
                elif issubclass(param_def["type"], float):
                    bounds[param_name] = (
                        param_def["bounds"][0],
                        param_def["bounds"][1],
                        float,
                    )
                else:
                    raise ValueError(
                        "Bounds can be defined only for numeric parameters."
                    )
            elif "values" in param_def:
                bounds[param_name] = list(param_def["values"])
            else:
                bounds[param_name] = (
                    None,
                    None,
                    param_def["type"],
                )
        # NOTE: user-defined bounds override defaults
        for param_name, param_value in user_defined_bounds.items():
            if isinstance(param_value, tuple):
                if any(isinstance(v, float) for v in param_value):
                    bounds[param_name] = (
                        param_value[0],
                        param_value[1],
                        float,
                    )
                    update_bounds(
                        method,
                        param_name,
                        param_value[0],
                        param_value[1],
                        None,
                    )
                elif any(isinstance(v, int) for v in param_value):
                    bounds[param_name] = (param_value[0], param_value[1], int)
                    update_bounds(
                        method,
                        param_name,
                        param_value[0],
                        param_value[1],
                        None,
                    )
                elif any(isinstance(v, str) for v in param_value):
                    bounds[param_name] = list(param_value)
                    update_bounds(
                        method, param_name, None, None, list(param_value)
                    )
                else:
                    raise TypeError(
                        f"Invalid bounds for parameter '{param_name}': {param_value}"
                    )
            elif isinstance(param_value, list):
                bounds[param_name] = param_value
                update_bounds(
                    method, param_name, None, None, param_value
                )                
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

    def _evaluate_params(self, trial) -> float:
        """
        Evaluate a set of hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            Current optimization trial.

        Returns
        -------
        float
            Negative metric value (since BayesianOptimization maximizes).
        """
        kwargs = self._suggest_arguments(trial)
        default_kwargs = self.reconstructor_kwargs.copy()
        default_kwargs.update(kwargs)
        try:
            log.debug("Evaluating parameters: %s", **default_kwargs)
            reconstructed = self.train_dset.reconstruct(
                target=self.val_dset.domain,
                method=self.method,
                **default_kwargs,
            )
            comparison = Comparison(reconstructed, self.val_dset)
            score = comparison.compute(self.metric.value)

            log.debug("Score for params %s: %f", kwargs, score)
            # NOTE: Return negative score for maximization
            score = self.scoring_callback(trial.number, kwargs, score)
            return score if np.isfinite(score) else DEFAULT_BAD_SCORE

        except (NotImplementedError, ValueError, TypeError) as e:
            log.warning("Error evaluating parameters %s: %s", kwargs, e)
            return DEFAULT_BAD_SCORE

    def restore_default_bounds(self) -> None:
        """Restore default bounds for all hyperparameters."""
        reconstructor_class = BaseReconstructor.get(self.method)
        hparams = reconstructor_class.get_hparams()
        for param_name, param_def in hparams.items():
            getattr(reconstructor_class, param_name).restore_default_bounds()

    @raise_if_not_installed("optuna")
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

        import optuna

        log.info("Starting Bayesian optimization for method '%s'", self.method)
        log.info("Bounds: %s", self.bounds)
        log.info("Using %d iterations", self.n_iters)
        sampler = optuna.samplers.GPSampler(
            seed=self.random_seed,
            deterministic_objective=True,
            n_startup_trials=self.n_startup_trials,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.n_startup_trials,
            n_warmup_steps=self.n_warmup_steps,
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.method}_study",
            load_if_exists=False,
        )

        study.optimize(
            self._evaluate_params,
            n_trials=self.n_iters,
            timeout=None,
            show_progress_bar=True,
        )

        log.info("Optimization completed. Best score: %f", study.best_value)
        log.info("Best parameters: %s", study.best_params)

        self.result = {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "metric_name": self.metric.value,
            "n_trials": len(study.trials),
            "method": self.method,
        }
        return self.result

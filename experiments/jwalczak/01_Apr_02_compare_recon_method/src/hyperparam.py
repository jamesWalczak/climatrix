from functools import partial
from typing import Callable

from bayes_opt import BayesianOptimization

import climatrix as cm


def find_hyperparameters(
    train_dset: cm.BaseClimatrixDataset,
    val_dset: cm.BaseClimatrixDataset,
    func: Callable[
        [cm.BaseClimatrixDataset, cm.BaseClimatrixDataset, dict], float
    ],
    bounds: dict[str, tuple],
    n_init_points: int = 30,
    n_iter: int = 200,
    seed: int = 0,
) -> None:
    func = partial(func, train_dset=train_dset, val_dset=val_dset)
    optimizer = BayesianOptimization(
        f=func,
        pbounds=bounds,
        random_state=seed,
    )
    optimizer.maximize(
        init_points=n_init_points,
        n_iter=n_iter,
    )
    return optimizer.max

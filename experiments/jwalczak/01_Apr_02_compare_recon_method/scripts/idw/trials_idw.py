"""
This module contains the code for the IDW trials.

All hyper-parameters were selected using constrained Bayesian
optimisation.

Bayesian optimisation output (init_points=30, n_iter=100)
{
    'target (RMSE)': np.float64(-1.4241189958579608), 
    'params': 
    {
        'k': np.float64(27.40201996616449), 
        'k_min': np.float64(17.348586061728497), 
        'power': np.float64(2.7965365027773164)
    }
}
"""
import xarray as xr
import climatrix as cm

TRIALS: int = 30

K: int = 27
POWER: float =  2.79
K_MIN: int = 17

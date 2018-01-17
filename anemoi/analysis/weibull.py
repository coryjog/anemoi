import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import scipy.optimize as spyopt
from scipy.special import gamma

### WEIBUL FITTING ###
def least_sq_fit(freqs, ws_bin_centers):
    '''Least squares fitting of parameters via data fitting to the distribution
    '''

    def weibull_least_sq_residuals(p, y, x):
        A, k = p
        residuals = y-k/A*(x/A)**(k-1.0)*np.exp(-(x/A)**k)
        return residuals

    params_init = [10, 2]
    lsq_A_k = spyopt.leastsq(weibull_least_sq_residuals, params_init, args=(freqs, ws_bin_centers))
    return lsq_A_k[0]

def euro_atlas_fit(ws_data):
    '''European Wind Atlas approach for calculating weibull parameters.
    This approach specifies the following contraints:
    - The total wind energy in the fitted weibull distribution must be equal
    to that of the observed distribution
    -The frequency of occurence of the wind speeds higher than the observed
    average speeds are the same for the two distributions, effectively placing
    more emphasis on the energetic wind speeds
    '''

    def estimate_weibull_k(ws_data, ws_mean, third_order_moment, prob_exceed_mean):
        k = np.exp(-(ws_mean/(third_order_moment/gamma(1.0+3.0/ws_data))**(1.0/3.0))**ws_data)-prob_exceed_mean
        return k

    third_order_moment = np.sum(ws_data**3.0)/len(ws_data)
    exceed_mean = ws_data > np.mean(ws_data)
    prob_exceed_mean = np.sum(exceed_mean, dtype=float)/len(ws_data)

    solve_k = spyopt.fsolve(estimate_weibull_k, x0=[2], args=(np.mean(ws_data), third_order_moment, prob_exceed_mean))
    k = np.round(solve_k[0], 3)
    A = np.round((third_order_moment/gamma(1+(3.0/k)))**(1.0/3.0), 3)
    return A, k

def return_weibull_params(mast, sensor=None, method='least_sq'):
        mast.is_mast_data_size_greater_than_zero()
        sensor = mast.check_and_return_mast_ws_sensor(sensor)
        ws_data = mast.return_sensor_data([sensor]).dropna().values

        x_smooth = np.linspace(0.0, 25.0, 100)
        ws_bins = np.arange(0.0, np.round(np.max(ws_data)))
        ws_bin_centers = np.add(ws_bins[1:],ws_bins[:-1])/2
        freqs, _ = np.histogram(ws_data, bins=ws_bins, normed=True)

        if method == 'WAsP':
            A, k = euro_atlas_fit(ws_data)
        elif method == 'least_sq':
            A, k = least_sq_fit(freqs, ws_bin_centers)

        A = np.round(A, 3)
        k = np.round(k, 3)
        return A, k

# def return_ws_column_freq_dist(wind_speed_data):
#     data_binned = wind_speed_data.round(0).melt()
#     data_binned['ones'] = 1
#     data_binned = data_binned.groupby(['variable', 'value']).count().unstack('variable')
#     data_binned.columns = data_binned.columns.get_level_values(-1).astype(int)
#     data_binned = data_binned.sort_index(axis=1).fillna(0)
#     data_binned = data_binned/data_binned.sum(axis=0)
#     return data_binned
import pandas as pd
import numpy as np
from scipy.stats import norm

def simulate_ts(
    freq = '30min',
    total_indexes = 4032,
    changepoint_indexes = [1000, 2000],
    changepoint_directions = ['up', 'down'],
    mean = 100,
    std = 5,
    level_shift = 0.2,
    weekend_ratio = 0.9,
    seed = 11,
    ):
    """
    Simulate time series data with level-shifts

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series

    total_indexes: int, default=4032
        Number of data points in the time series
    
    changepoint_indexes: list of int, default=[1000, 2000]
        List containing indexes of changepoints
    
    changepoint_directions: list of str, default=['up', 'down']
        List containing directions of changepoints

    mean: float, default=100
        Mean of the random Gaussian distribution

    std: float, default=100
        Standard deviation of the random Gaussian distribution

    level_shift: float, default=100
        Degree of level shift

    weekend_ratio: float, default=0.9
        Proportion of weekend profile to weekday profile.
        Set weekend_ratio=1 to remove weekly seasonality.

    seed: int, default=11
        Random seed value for generating random Gaussian distribution.

    Returns
    -------
    df: pandas.DataFrame of shape (total_indexes, 1)
        Univariate time series simulated according to specified parameters
    """

    if pd.Series(changepoint_indexes).apply(lambda x: x >= total_indexes).any():
        raise ValueError('changepoint index must be smaller than total_indexes')
    if pd.Series(changepoint_directions).apply(lambda x: x not in ['up', 'down']).any():
        raise ValueError('changepoint direction must be "up" or "down"')
    if len(changepoint_indexes) != len(changepoint_directions):
        raise ValueError('changepoint_indexes and changepoint_directions must have the same length')

    periods = int(pd.Timedelta('1d')/pd.Timedelta(freq))
    np.random.seed(seed)
    y_value = norm.rvs(mean, std, total_indexes)

    # simulate daily seasonality
    for j in range(0, len(y_value)-periods+1, periods):
        increasing_start = int(pd.Timedelta('8h')/pd.Timedelta(freq))
        increasing_end = int(pd.Timedelta('13h')/pd.Timedelta(freq))
        for i in range(increasing_start, increasing_end):
            y_value[j+i] += (i-increasing_start)*10

        decreasing_end = int(pd.Timedelta('20h')/pd.Timedelta(freq))
        for i in range(increasing_end, decreasing_end):
            y_value[j+i] += (decreasing_end-i)*10

    # simulate weekly seasonality
    for idx in range(0, len(y_value)-periods*7, periods*7):
        for i in range(periods*5, periods*6):
            y_value[idx+i] *= weekend_ratio
        for i in range(periods*6, periods*7):
            y_value[idx+i] *= weekend_ratio

    # simulate level-shift
    for i in range(len(changepoint_indexes)):
        for j in range(changepoint_indexes[i], total_indexes):
            if changepoint_directions[i] == 'up':
                y_value[j] += level_shift * y_value[j]
            else:
                y_value[j] -= level_shift * y_value[j]

    df = pd.DataFrame({'y': y_value})
    df.index = pd.date_range(start='2020-01-06', periods=len(df), freq='30T')
    df.index.name = 'ts'

    return df

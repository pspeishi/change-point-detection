import pandas as pd
import numpy as np
import datetime
from sklearn.base import BaseEstimator, TransformerMixin

def remove_duplicate_rows(df):
    """Returns df with duplicates removed"""    
    idx_name = df.index.name

    return df.reset_index().drop_duplicates().set_index(idx_name)

def median_imputation(df, **kwargs):
    ycol = kwargs['ycol']
    median_profile = kwargs['median_profile']
    freq = kwargs['freq']
    na_th = kwargs['na_th']

    if df['ts'].max() - df['ts'].min() + pd.Timedelta(freq) >= pd.Timedelta(na_th):
        df[ycol] = df['ts'].dt.time.map(median_profile)

    return df[[ycol]]

def fill_na(df, ycol, median_profile, freq, na_th):
    df.reset_index(inplace=True)
    na_idx = df[ycol].isnull()
    df['count'] = df[ycol].notnull().cumsum()
    df1 = df[na_idx].copy()
    df.loc[na_idx, ycol] = df1.groupby('count').apply(median_imputation, ycol=ycol, median_profile=median_profile, freq=freq, na_th=na_th)
    df.set_index('ts', inplace=True)
    df[ycol] = pd.to_numeric(df[ycol], errors='coerce')
    df[ycol] = df[ycol].interpolate(method='linear',limit_direction = 'both')
    df.drop(columns='count', inplace=True)

    return df

def get_median_profile(df, ycol):
    """Calculates the median profiles for each time index of the day"""
    df['time'] = df.index.time
    median_profile = df.groupby('time')[ycol].median().to_dict()

    return median_profile

def get_outlier_profile(df, ycol, iqr_coeff):
    """Calculates the outlier profiles for each time index of the day for each month"""
    df['time'] = df.index.time
    
    q1 = df.groupby('time')[ycol].apply(lambda x: np.quantile(x, 0.25))
    q3 = df.groupby('time')[ycol].apply(lambda x: np.quantile(x, 0.75))
    iqr = q3 - q1
    df_stats = df.groupby('time').agg(['median'])[ycol]
    df_stats['lower_bound'] = q1 - iqr_coeff * iqr
    df_stats['upper_bound'] = q3 + iqr_coeff * iqr

    return df_stats[['median', 'lower_bound', 'upper_bound']].to_dict('index')

def is_outlier_helper(y, stats):
    """Helper function for is_outlier function"""
    if y < stats['lower_bound']:
        return True
    elif y > stats['upper_bound']:
        return True
    else:
        return False

def is_outlier(group, ycol, outlier_profile):
    """Labels data points lower than lower_bound or higher than upper_bound as outliers"""
    ts = group['time'].values[0]
    stats = outlier_profile[ts]
    y = group[ycol].values
    outlier = list(map(lambda x: is_outlier_helper(x, stats), y))

    return outlier

def replace_outlier_helper(y, stats):
    """Helper function for replace_outlier function"""
    if y <= 0:
        return stats['median']
    elif y < stats['lower_bound']:
        return stats['median']
    elif y > stats['upper_bound']:
        return stats['median']
    else:
        return y  

def replace_outlier(group, ycol, outlier_profile):
    """Median imputation for all outliers"""
    ts = group['time'].values[0]
    stats = outlier_profile[ts]
    y = group[ycol].values # array of ycol values
    outlier = list(map(lambda x: replace_outlier_helper(x, stats), y))

    return outlier

def handle_outlier(df, outlier_profile, ycol):
    df['time'] = df.index.time
    
    outlier_grouped = df.groupby(['time']).apply(lambda group: is_outlier(group, ycol, outlier_profile))
    y_grouped = df.groupby(['time']).apply(lambda group: replace_outlier(group, ycol, outlier_profile))

    y_ls = []
    outlier_ls = []
    for j in range(len(y_grouped[-1])):
        for i in range(len(y_grouped.index)):
            y_ls.append(y_grouped[i][j])
            outlier_ls.append(outlier_grouped[i][j])

    df['new_y'] = pd.Series(data=y_ls, index=df.index).sort_index().values
    df['outlier'] = pd.Series(data=outlier_ls, index=df.index).sort_index().values

    df1 = df[['new_y']]
    df1 = df1.rename(columns={'new_y': ycol})
    outliers = df[df['outlier']].index
    
    return df1, outliers

def check_missing_gap(df, freq, missing_gap_threshold, ycol):
    """Return start index of 24h missing gap if any, else return None"""
    min_periods = int(pd.Timedelta(missing_gap_threshold)/pd.Timedelta(freq))
    bool_df = df.isna()
    count = 0
    first_na = True
    for i in range(len(bool_df)):
        if bool_df[ycol][i]:
            count += 1
            if first_na:
                missing_gap_start = bool_df.index[i]
                first_na = False
            if count >= min_periods:
                return missing_gap_start
        else:
            count = 0
            first_na = True
    return None

class Preprocessor(BaseEstimator, TransformerMixin):
    """Preprocesses a univariate time series dataframe into a time series of the specified frequency

    1) Remove duplicated rows
    2) Check for missing gap of length >= missing_gap_threshold
        - If missing gap is present, preprocessing is halted and current iteration in the main Detector is skipped.
        - If no missing gap:
            - Replace missing values with median profile of the corresponding time index
            - Find and replace outliers with median profile of the corresponding time index
            - If ewm=True, time series will be exponentially weighted.

    Parameters
    ----------
    freq: str, default='30min'
        Resample frequency of time series. 

    na_th: str, default='2h'
        Threshold of data length for median imputation. 

    agg: {'mean', 'median', 'std', 'min', or 'max'), default='mean'
        Aggregation function for resampling.
    
    missing_gap_threshold: str, default='1d'
        Missing gaps >= missing_gap_threshold will be reported.
    
    iqr_coeff: float, default=5
        Used to obtain the upper and lower bound profiles of each time index.
        Values beyond these bounds are considered outliers.

    ewm: bool, default=True
        If True, time series will be exponentially weighted.

    alpha: float, default=0.9
        Smoothing factor for exponential smoothing. Only applicable when ewm=True.

    Attributes
    ----------
    missing_gap: pandas.Timestamp or None
        Starting time index of missing gap

    ycol: str
        Column name of the original dataframe
    
    median_profile: dict
        Median profiles of each time index

    outlier_profile: dict
        Upper and lower bound profiles of each time index

    """
    def __init__(self, freq='30min', na_th='2h', missing_gap_threshold='1d', iqr_coeff=5, ewm=True, alpha=0.9):
        self.freq = freq
        self.na_th = na_th
        self.missing_gap_threshold = missing_gap_threshold
        self.iqr_coeff = iqr_coeff
        self.ewm = ewm
        self.alpha = alpha
        self.outliers = []

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X):
        """ Pre-processes the dataframe using the fitted outlier profiles and median profiles. 
        
        Parameters
        ----------
        X :  pd.DataFrame of shape (n_samples, 1)
             The univariate data to process and convert into time series of specified freq.
       
        Returns
        -------
        X_tr : pd.DataFrame shape (n_samples, 1)
            Time Series Dataframe of specified frequency
        """
        X = X.copy()
        if len(X.columns) > 1:
            raise ValueError("Input data is not univariate")

        self.ycol = X.columns[0]

        X = remove_duplicate_rows(X.copy(deep=True))
        if sum(X.index.duplicated()) != 0:
            raise Exception("Rows with duplicate datetimes detected")

        self.missing_gap = check_missing_gap(X, self.freq, self.missing_gap_threshold, self.ycol)

        if self.missing_gap == None:
            self.median_profile = get_median_profile(X, self.ycol)
            self.outlier_profile = get_outlier_profile(X, self.ycol, self.iqr_coeff)

        if self.missing_gap == None:
            df = fill_na(X, self.ycol, self.median_profile, self.freq, self.na_th)
            df = df[[self.ycol]]
            if self.iqr_coeff != None:
                X, outliers = handle_outlier(X, self.outlier_profile, self.ycol)
                self.outliers = outliers
            if self.ewm:
                X = X.ewm(alpha=self.alpha).mean()

        return X
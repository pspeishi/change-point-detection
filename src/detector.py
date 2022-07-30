import pandas as pd
import numpy as np
from tqdm import tqdm
import ruptures as rpt
import matplotlib.pyplot as plt
from scipy.stats import chi2

from preprocessor import Preprocessor

TESTS_DICT = {
    "n_day_rule": True,
    "log_likelihood_ratio_test": True,
    "relative_change_test": True,
    "magnitude_test": True,
    "absolute_change_test": True,
}

DEFAULT_ARGS = {
    "freq": '30min',
    "na_th": '2h',
    "preprocessing_window": 'full',
    "cpd_algorithm": 'dynp',
    "cpd_cost": 'l2',
    "agg": 'mean',
    "missing_gap_threshold": '1d',
    "iqr_coeff": 5,
    "ewm": True,
    "alpha": 0.9,
    "historical_window": '14d',
    "scan_window": '7d',
    "buffer_window": '7d',
    "step": '1d',
    "tests": TESTS_DICT,
    "n_day_rule_threshold": '1d',
    "llr_threshold": 0.01,
    "magnitude_quantile": 0.05,
    "magnitude_quantile_direction": "any",
    "magnitude_median_direction": True,
    "magnitude_ratio": 0.2,
    "magnitude_ratio_multiplier": 0.75,
    "magnitude_comparable_day": 0.5,
    "perc_change_threshold": 0.1,
    "abs_change_threshold": 0.01,
}

CPD_ALGORITHMS = ['dynp', 'binseg', 'bottomup']

def _get_arg(name, **kwargs):
    return kwargs.get(name, DEFAULT_ARGS[name])

class Detector:
    """ Detector to find changepoints and related statistics on a rolling window basis

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
        
    na_th: str, default='2h'
        Threshold of data length for median imputation

    preprocessing_window: {'full', 'separate'}, default='full'
        If 'full', each rolling window will be preprocessed as a whole.
        If 'separate', historical, scan and buffer window will be preprocessed separately.
    
    cpd_algorithm: {'dynp', 'binseg', 'bottomup'}, default='dynp'
        Algorithm for changepoint detection

    cpd_cost: str, default='l2'
        Cost function for changepoint detection

    agg: {'mean', 'median', 'std', 'min', or 'max'), default='mean'
        Aggregation function for resampling

    missing_gap_threshold: str, default='1d'
        Missing gaps >= missing_gap_threshold will be reported
        
    iqr_coeff: float, default=5
        Used to obtain the upper and lower bound profiles of each time index.
        Values beyond these bounds are considered outliers.

    ewm: bool, default=True
        If True, time series will be exponentially weighted.

    alpha: float, default=0.9
        Smoothing factor for exponential smoothing. Ignored if ewm=False.

    historical_window: str, default='14d'
        Number of days in historical window.

    scan_window: str, default='7d'
        Number of days in scan window.

    buffer_window: str, default='7d'
        Number of days in buffer window.

    step: str, default='1d'
        Number of days the window rolls ahead for each iteration.

    tests: dict, default=TESTS_DICT
        Specify which tests to be conducted on the changepoints.

    n_day_rule_threshold: str, default='1d'
        Minimum number of days between changepoint and scan window boundaries.

    llr_threshold: float, default=0.01
        Significance level for log-likelihood ratio tests.

    magnitude_quantile: float, default=0.05
        Percentile value used in magnitude test.

    magnitude_quantile_direction: {'top', 'bottom', 'median', 'mean', 'all', 'any'}, default='any'
        Direction of percentile value used in magnitude test.

    magnitude_median_direction: bool, default=True
        If True, direction is considered when using median magnitude test.
    
    magnitude_ratio: float, default=0.2
        Multiplied with magnitude_ratio_multiplier to generate the comparable ratio to be used magnitude test.

    magnitude_ratio_multiplier: float, default=0.75
        Multiplied with magnitude_ratio to generate the comparable ratio to be used magnitude test.

    magnitude_comparable_day: float, default=0.5
        Minimum proportion of sliding windows with percentile value > magnitude_ratio * magnitude_ratio_multiplier to pass magnitude test.

    perc_change_threshold: float, default=0.1
        Threshold for relative change tests.
    
    abs_change_threshold: float, default=0.1
        Threshold for absolute change tests.

    Attributes
    ----------
    changepoints: list
        List of time indexes of detected changepoints that passed all condition checks
    
    missing_gaps: list
        List of starting time indexes of missing gaps
    """
    def __init__(self, **kwargs):
        self.freq = _get_arg("freq", **kwargs)
        self.periods = int(pd.Timedelta('1d') / self.freq)
        self.na_th = _get_arg("na_th", **kwargs)
        self.preprocessing_window = _get_arg("preprocessing_window", **kwargs)
        self.cpd_algorithm = _get_arg("cpd_algorithm", **kwargs)
        self.cpd_cost = _get_arg("cpd_cost", **kwargs)
        self.agg = _get_arg("agg", **kwargs)
        self.missing_gap_threshold = _get_arg("missing_gap_threshold", **kwargs)
        self.iqr_coeff = _get_arg("iqr_coeff", **kwargs)
        self.ewm = _get_arg("ewm", **kwargs)
        self.alpha = _get_arg("alpha", **kwargs)
        self.historical_window = _get_arg("historical_window", **kwargs) 
        self.scan_window = _get_arg("scan_window", **kwargs) 
        self.buffer_window = _get_arg("buffer_window", **kwargs) 
        self.historical_window_periods = int(pd.Timedelta(self.historical_window) / pd.Timedelta(self.freq))
        self.scan_window_periods = int(pd.Timedelta(self.scan_window) / pd.Timedelta(self.freq))
        self.buffer_window_periods = int(pd.Timedelta(self.buffer_window) / pd.Timedelta(self.freq))
        self.step = _get_arg("step", **kwargs) 
        self.tests = _get_arg("tests", **kwargs)
        self.n_day_rule_threshold = int(pd.Timedelta(_get_arg("n_day_rule_threshold", **kwargs)) / pd.Timedelta(self.freq))
        self.llr_threshold = _get_arg("llr_threshold", **kwargs)
        self.magnitude_quantile = _get_arg("magnitude_quantile", **kwargs)
        self.magnitude_quantile_direction = _get_arg("magnitude_quantile_direction", **kwargs)
        self.magnitude_median_direction = _get_arg("magnitude_median_direction", **kwargs)
        self.magnitude_ratio = _get_arg("magnitude_ratio", **kwargs)
        self.magnitude_ratio_multiplier = _get_arg("magnitude_ratio_multiplier", **kwargs)
        self.magnitude_comparable_day = _get_arg("magnitude_comparable_day", **kwargs)
        self.perc_change_threshold = _get_arg("perc_change_threshold", **kwargs)
        self.abs_change_threshold = _get_arg("abs_change_threshold", **kwargs)
        self.changepoints = [] # changepoints that passed all tests
        self.missing_gaps = [] # changepoints due to missing gap
        self.changepoints_and_gaps = [] # changepoints that passed all tests or due to missing gap
        self.fit_predict_called = False

        # for tracking of all changepoints 
        self.changepoints_all = []
        self.scan_window_start_all = []
        self.scan_window_end_all = []
        self.n_day_rule_values_all = []
        self.llr_7d_pvalues_all = []
        self.llr_backward_pvalues_all = []
        self.llr_forward_pvalues_all = []
        self.mag_values_all = []
        self.perc_change_7d_values_all = []
        self.perc_change_backward_values_all = []
        self.perc_change_forward_values_all = []
        self.perc_change_hist_buffer_values_all = []
        self.abs_change_7d_values_all = []
        self.abs_change_backward_values_all = []
        self.abs_change_forward_values_all = []
        self.directions_all = []
        self.passed_all_tests = []
        self.duplicate = []
        self.missing_gap_boolean = []

        # raise errors for inappropriate parameters
        if self.cpd_algorithm not in CPD_ALGORITHMS:
            raise ValueError(f'Supported cpd algorithms {CPD_ALGORITHMS}')
        if self.historical_window_periods < self.scan_window_periods:
            raise ValueError('Length of historical window must be larger than length of scan window')

    def _detect_changepoint(self, curr_df):
        """
        Runs CPD algorithm using AMOC concept

        Parameters
        ----------
        curr_df: pandas.DataFrame of shape (n_samples, 1)
            Univariate time series dataframe that has been preprocessed
        
        Returns
        -------
        cp: int
            Index of changepoint within scan window
        """        
        scan_df = curr_df[self.historical_window_periods : self.historical_window_periods + self.scan_window_periods]

        # run cpd algorithm to detect changepoint
        if self.cpd_algorithm == 'dynp':
            algo = rpt.Dynp(model=self.cpd_cost).fit(scan_df.values)
        elif self.cpd_algorithm == 'binseg':
            algo = rpt.Binseg(model=self.cpd_cost).fit(scan_df.values)
        else:
            algo = rpt.BottomUp(model=self.cpd_cost).fit(scan_df.values)
        try:
            cp = algo.predict(n_bkps=1)[0]
            return cp
        except:
            return None

    def _n_day_rule(self, changepoint):
        """
        Return True if changepoint is at least n days from start and end of scan window

        Parameters
        ----------
        changepoint: int
            Index of changepoint within scan window
        
        Returns
        -------
        True or False
        """
        if changepoint < self.n_day_rule_threshold or self.scan_window_periods - changepoint < self.n_day_rule_threshold:
            return False
        else:
            return True

    def _get_llr_pvalue(self, llr):
        """
        Calculate log likelihood ratio p-value

        Parameters
        ----------
        llr: float
            Log likelihood ratio
        
        Returns
        -------
        pvalue: float
            Log likelihood ratio p-value
        """
        return 1 - chi2.cdf(llr, 2)

    def _get_llr(self, ts, mu0, mu1, changepoint):
        """
        Calculate log likelihood ratio of two time series segment

        Parameters
        ----------
        ts: pandas.DataFrame of shape (n_samples, 1)
            Appended time series of the two segments to be compared

        mu0: float
            Mean of first time series segment

        mu1: float
            Mean of second time series segment
        
        changepoint: int
            Index of appended time series that separates both segments
        
        Returns
        -------
        llr: float
            Log likelihood ratio of the two time series segment
        """
        scale = np.sqrt(
            (
                np.sum((ts[:changepoint] - mu0) ** 2)
                + np.sum((ts[changepoint+1:] - mu1) ** 2)
            )
            / (len(ts) - 2)
        ).values[0]

        ts_without_cp = ts[:changepoint].append(ts[changepoint+1:])
        mu_tilde, sigma_tilde = np.mean(ts_without_cp).values[0], np.std(ts_without_cp).values[0]

        if scale == 0:
            scale = sigma_tilde

        llr = -2 * (
            self._log_llr(ts[:changepoint], mu_tilde, sigma_tilde, mu0, scale)
            + self._log_llr(ts[changepoint+1:], mu_tilde, sigma_tilde, mu1, scale)
        )
        return llr

    def _log_llr(self, x, mu0, sigma0, mu1, sigma1):
        """
        Helper function to calculate log likelihood ratio.
        This function calculate the log likelihood ratio of two Gaussian
        distribution log(l(0)/l(1)).

        Parameters
        ----------
        x: pandas.DataFrame of shape (n_samples, 1)
            Time series of one segment

        mu0: float
            Mean of model0

        sigma0: float
            Standard deviation of model0

        mu1: float
            Mean of model1

        sigma1: float
            Standard deviation of model1
        
        Returns
        -------
        llr: float
            Value of log likelihood ratio
        """
        return np.sum(
            np.log(sigma1 / sigma0)
            + 0.5 * (((x - mu1) / sigma1) ** 2 - ((x - mu0) / sigma0) ** 2)
        ).values[0]

    def _get_mag(self, scan_df, curr_df, direction):
        """
        Calculate proportion of sliding windows in historical window that passes the magnitude test

        Paramters
        ---------
        scan_df: pandas.DataFrame of shape (length of scan window, 1)
            Time series within scan window

        curr_df: pandas.DataFrame of shape (length of rolling window, 1)
            Time series within rolling window (historical window + scan window + buffer window)

        direction: {'up' or 'down'}
            Direction of changepoint

        Returns
        -------
        (count/total): float
            Proportion of sliding windows in historical window that passes the magnitude test
        """
        p_scan_top = np.quantile(scan_df.values, 1-self.magnitude_quantile)
        p_scan_median = np.quantile(scan_df.values, 0.5)
        p_scan_mean = np.mean(scan_df.values)
        p_scan_bottom = np.quantile(scan_df.values, self.magnitude_quantile)
        count = 0
        total = 0
        for i in range(self.historical_window_periods, self.scan_window_periods-self.periods, -self.periods):
            curr = curr_df[i - self.scan_window_periods : i]
            p_bottom = np.quantile(curr.values, self.magnitude_quantile)
            p_median = np.quantile(curr.values, 0.5)
            p_mean = np.mean(curr.values)
            p_top = np.quantile(curr.values, 1-self.magnitude_quantile)
            total += 1
            if direction == 'up':
                compare_top = (p_scan_top - p_top) / (p_top+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
                compare_bottom = (p_scan_bottom - p_bottom) / (p_bottom+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
                compare_median = (p_scan_median - p_median) / (p_median+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
                compare_mean = (p_scan_mean - p_mean) / (p_mean+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
            else:
                compare_top = (p_top - p_scan_top) / (p_top+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
                compare_bottom = (p_bottom - p_scan_bottom) / (p_bottom+0.00001)  > self.magnitude_ratio_multiplier * self.magnitude_ratio
                compare_median = (p_median - p_scan_median) / (p_median+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
                compare_mean = (p_mean - p_scan_mean) / (p_mean+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
            
            if self.magnitude_median_direction == False:
                compare_median = abs(p_scan_median - p_median) / (p_median+0.00001) > self.magnitude_ratio_multiplier * self.magnitude_ratio
            
            if self.magnitude_quantile_direction == 'top' and compare_top:
                count += 1
            if self.magnitude_quantile_direction == 'bottom' and compare_bottom:
                count += 1
            if self.magnitude_quantile_direction == 'median' and compare_median:
                count += 1
            if self.magnitude_quantile_direction == 'mean' and compare_mean:
                count += 1
            if self.magnitude_quantile_direction == 'all' and all([compare_top, compare_bottom, compare_median, compare_mean]):
                count += 1
            if self.magnitude_quantile_direction == 'any' and any([compare_top, compare_bottom, compare_median, compare_mean]):
                count += 1
                
        return count/total

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        """
        Preprocesses data, detects changepoints and conducts condition checks on a rolling window basis

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            Time series data to find changepoints
        y : None
            Ignored.
        
        Returns
        -------
        changepoints: list
            List of detected changepoints that passed all condition checks

        """
        df = X.copy()
        df = df.resample(self.freq).agg(self.agg)
        idx_new = list(pd.date_range(start=pd.Timestamp(df.index[0].date()), end=df.index[-1], freq=self.freq))
        df = df.reindex(idx_new)
        self.df = df
        self.fit_predict_called = True
        window_periods = self.historical_window_periods + self.scan_window_periods + self.buffer_window_periods
        for i in tqdm(range(0, len(df) - window_periods + 1, int(pd.Timedelta(self.step) / pd.Timedelta(self.freq)))):
            # preprocess current window of data
            curr_df = df.iloc[i:i+window_periods].copy()
            if self.preprocessing_window == 'full':
                preprocessor = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
                curr_df = preprocessor.fit_transform(curr_df)
                curr_missing_gap = preprocessor.missing_gap
            elif self.preprocessing_window == 'separate':
                preprocessor_hist = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
                hist_df = preprocessor_hist.fit_transform(curr_df[:self.historical_window_periods])
                preprocessor_scan = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
                scan_df = preprocessor_scan.fit_transform(curr_df[self.historical_window_periods:self.historical_window_periods+self.scan_window_periods])
                preprocessor_buffer = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
                buffer_df = preprocessor_buffer.fit_transform(curr_df[-self.buffer_window_periods:])
                curr_df = hist_df.append(scan_df.append(buffer_df))
                if preprocessor_hist.missing_gap:
                    curr_missing_gap = preprocessor_hist.missing_gap
                elif preprocessor_scan.missing_gap:
                    curr_missing_gap = preprocessor_scan.missing_gap
                else:
                    curr_missing_gap = preprocessor_buffer.missing_gap
            else:
                raise ValueError('preprocessing_window must be "full" or "separate"')

            if curr_missing_gap:
                if len(self.missing_gaps) == 0:
                    self.missing_gaps.append(curr_missing_gap)
                    self.changepoints_all.append(curr_missing_gap)
                    self.changepoints_and_gaps.append(curr_missing_gap)
                elif curr_missing_gap != self.missing_gaps[-1] and curr_missing_gap != curr_df.index[0]:
                    self.missing_gaps.append(curr_missing_gap)
                    self.changepoints_all.append(curr_missing_gap)
                    self.changepoints_and_gaps.append(curr_missing_gap)
                else:
                    self.changepoints_all.append(self.missing_gaps[-1])

                # for tracking of changepoints due to missing gap
                self.scan_window_start_all.append(curr_df.index[0] + self.historical_window_periods * pd.Timedelta(self.freq))
                self.scan_window_end_all.append(curr_df.index[0] + (self.historical_window_periods + self.scan_window_periods - 1) * pd.Timedelta(self.freq))
                self.n_day_rule_values_all.append(None)
                self.llr_7d_pvalues_all.append(None)
                self.llr_backward_pvalues_all.append(None)
                self.llr_forward_pvalues_all.append(None)
                self.mag_values_all.append(None)
                self.directions_all.append(None)
                self.perc_change_7d_values_all.append(None)
                self.perc_change_backward_values_all.append(None)
                self.perc_change_forward_values_all.append(None)
                self.perc_change_hist_buffer_values_all.append(None)
                self.abs_change_7d_values_all.append(None)
                self.abs_change_backward_values_all.append(None)
                self.abs_change_forward_values_all.append(None)
                self.passed_all_tests.append(False)
                self.duplicate.append(None)
                self.missing_gap_boolean.append(True)
                continue

            # detect changepoint
            cp = self._detect_changepoint(curr_df)
            if cp == None:
                # for tracking of changepoints returned as None
                self.changepoints_all.append(None)
                self.scan_window_start_all.append(curr_df.index[0] + self.historical_window_periods * pd.Timedelta(self.freq))
                self.scan_window_end_all.append(curr_df.index[0] + (self.historical_window_periods + self.scan_window_periods - 1) * pd.Timedelta(self.freq))
                self.n_day_rule_values_all.append(None)
                self.llr_7d_pvalues_all.append(None)
                self.llr_backward_pvalues_all.append(None)
                self.llr_forward_pvalues_all.append(None)
                self.mag_values_all.append(None)
                self.directions_all.append(None)
                self.perc_change_7d_values_all.append(None)
                self.perc_change_backward_values_all.append(None)
                self.perc_change_forward_values_all.append(None)
                self.perc_change_hist_buffer_values_all.append(None)
                self.abs_change_7d_values_all.append(None)
                self.abs_change_backward_values_all.append(None)
                self.abs_change_forward_values_all.append(None)
                self.passed_all_tests.append(False)
                self.duplicate.append(None)
                self.missing_gap_boolean.append(False)
                continue

            # n_day_rule
            if self.tests["n_day_rule"]:
                if not self._n_day_rule(cp):
                    # for tracking of changepoints that fail n-day rule
                    self.changepoints_all.append(curr_df.index[0] + (self.historical_window_periods + cp) * pd.Timedelta(self.freq))
                    self.scan_window_start_all.append(curr_df.index[0] + self.historical_window_periods * pd.Timedelta(self.freq))
                    self.scan_window_end_all.append(curr_df.index[0] + (self.historical_window_periods + self.scan_window_periods - 1) * pd.Timedelta(self.freq))
                    self.n_day_rule_values_all.append('Fail')
                    self.llr_7d_pvalues_all.append(None)
                    self.llr_backward_pvalues_all.append(None)
                    self.llr_forward_pvalues_all.append(None)
                    self.mag_values_all.append(None)
                    self.directions_all.append(None)
                    self.perc_change_7d_values_all.append(None)
                    self.perc_change_backward_values_all.append(None)
                    self.perc_change_forward_values_all.append(None)
                    self.perc_change_hist_buffer_values_all.append(None)
                    self.abs_change_7d_values_all.append(None)
                    self.abs_change_backward_values_all.append(None)
                    self.abs_change_forward_values_all.append(None)
                    self.passed_all_tests.append(False)
                    self.duplicate.append(None)
                    self.missing_gap_boolean.append(False)
                    continue

            ## 7d before and after changepoint ##
            df_7d = curr_df[self.historical_window_periods + cp - self.periods*7 : self.historical_window_periods + cp + self.periods*7 + 1]
            mu0 = np.mean(df_7d[:len(df_7d)//2]).values[0]
            mu1 = np.mean(df_7d[len(df_7d)//2+1:]).values[0]

            # log-likelihood test
            llr_7d = self._get_llr(df_7d, mu0, mu1, len(df_7d)//2)
            pvalue_7d = self._get_llr_pvalue(llr_7d)

            # get change direction
            if mu1 - mu0 > 0:
                direction = 'up'
            else:
                direction = 'down'

            # abs change and perc change
            abs_change_7d = abs(mu1 - mu0)
            perc_change_7d = abs_change_7d/(mu0+0.00001)


            ## 7d apart in backward direction ##
            backward_df = curr_df[self.historical_window_periods + cp - 7*self.periods + 1 : self.historical_window_periods + self.scan_window_periods - 7*self.periods].append(curr_df[self.historical_window_periods + cp : self.historical_window_periods + self.scan_window_periods])
            mu0 = np.mean(backward_df[:len(backward_df)//2]).values[0]
            mu1 = np.mean(backward_df[len(backward_df)//2+1:]).values[0]

            # log-likelihood test
            llr_backward = self._get_llr(backward_df, mu0, mu1, len(backward_df)//2)
            pvalue_backward = self._get_llr_pvalue(llr_backward)

            # abs change and perc change
            if direction == 'up':
                abs_change_backward = mu1 - mu0
            else:
                abs_change_backward = mu0 - mu1
            perc_change_backward = abs_change_backward/(mu0+0.00001)


            ## 7d apart in forward direction ##
            forward_df = curr_df[self.historical_window_periods : self.historical_window_periods + cp + 1].append(curr_df[self.historical_window_periods + 7*self.periods : self.historical_window_periods + cp + 7*self.periods])
            mu0 = np.mean(forward_df[:len(forward_df)//2]).values[0]
            mu1 = np.mean(forward_df[len(forward_df)//2+1:]).values[0]

            # log-likelihood test
            llr_forward = self._get_llr(forward_df, mu0, mu1, len(forward_df)//2)
            pvalue_forward = self._get_llr_pvalue(llr_forward)

            # abs change and perc change
            if direction == 'up':
                abs_change_forward = mu1 - mu0
            else:
                abs_change_forward = mu0 - mu1
            perc_change_forward = abs_change_forward/(mu0+0.00001)

            # historical vs buffer relative change
            historical_mean = curr_df[:self.historical_window_periods].values.mean()
            buffer_mean = curr_df[-self.buffer_window_periods:].values.mean()
            if direction == 'up':
                abs_change_hist_buffer = buffer_mean - historical_mean
            else:
                abs_change_hist_buffer =  historical_mean - buffer_mean
            perc_change_hist_buffer = abs_change_hist_buffer/(historical_mean+0.00001)

            # magnitude test
            scan_df = curr_df[self.historical_window_periods : self.historical_window_periods + self.scan_window_periods]
            mag_value = self._get_mag(scan_df, curr_df, direction)

            # tracking of all changepointss
            self.changepoints_all.append(curr_df.index[0] + (self.historical_window_periods + cp) * pd.Timedelta(self.freq))
            self.scan_window_start_all.append(curr_df.index[0] + self.historical_window_periods * pd.Timedelta(self.freq))
            self.scan_window_end_all.append(curr_df.index[0] + (self.historical_window_periods + self.scan_window_periods - 1) * pd.Timedelta(self.freq))
            self.n_day_rule_values_all.append('Pass')
            self.llr_7d_pvalues_all.append(pvalue_7d)
            self.llr_backward_pvalues_all.append(pvalue_backward)
            self.llr_forward_pvalues_all.append(pvalue_forward)
            self.mag_values_all.append(mag_value)
            self.perc_change_7d_values_all.append(perc_change_7d)
            self.perc_change_backward_values_all.append(perc_change_backward)
            self.perc_change_forward_values_all.append(perc_change_forward)
            self.perc_change_hist_buffer_values_all.append(perc_change_hist_buffer)
            self.abs_change_7d_values_all.append(abs_change_7d)
            self.abs_change_backward_values_all.append(abs_change_backward)
            self.abs_change_forward_values_all.append(abs_change_forward)
            self.directions_all.append(direction)
            self.missing_gap_boolean.append(False)

            # test results
            llr_7d_res = pvalue_7d < self.llr_threshold
            llr_backward_res = pvalue_backward < self.llr_threshold
            llr_forward_res = pvalue_forward < self.llr_threshold
            mag_test_res = mag_value >= self.magnitude_comparable_day
            perc_change_7d_res = perc_change_7d >= self.perc_change_threshold
            perc_change_backward_res = perc_change_backward >= self.perc_change_threshold
            perc_change_forward_res = perc_change_forward >= self.perc_change_threshold
            perc_change_hist_buffer_res = perc_change_hist_buffer >= self.perc_change_threshold
            abs_change_7d_res = abs_change_7d >= self.abs_change_threshold
            abs_change_backward_res = abs_change_backward >= self.abs_change_threshold
            abs_change_forward_res = abs_change_forward >= self.abs_change_threshold

            test_ls = []
            if self.tests["log_likelihood_ratio_test"]:
                test_ls.append(llr_7d_res)
                test_ls.append(llr_backward_res)
                test_ls.append(llr_forward_res)
            if self.tests["relative_change_test"]:
                test_ls.append(perc_change_7d_res)
                test_ls.append(perc_change_backward_res)
                test_ls.append(perc_change_forward_res)
                test_ls.append(perc_change_hist_buffer_res)
            if self.tests["absolute_change_test"]:
                test_ls.append(abs_change_7d_res)
                test_ls.append(abs_change_backward_res)
                test_ls.append(abs_change_forward_res)
            if self.tests["magnitude_test"]:
                test_ls.append(mag_test_res)

            res = all(test_ls)
            if res:
                self.passed_all_tests.append(True)
                curr_cp = curr_df.index[0] + (self.historical_window_periods + cp) * pd.Timedelta(self.freq)
                
                # check if curr_cp is within 24h of previously detected changepoints
                no_duplicate = True
                for changepoint in self.changepoints:
                    if abs(curr_cp - changepoint) < pd.Timedelta('1d'):
                        no_duplicate = False
                        break
                if no_duplicate:
                    self.changepoints.append(curr_cp)
                    self.duplicate.append(False)
                    self.changepoints_and_gaps.append(curr_cp)
                else:
                    self.duplicate.append(True)

            else:
                self.passed_all_tests.append(False)
                self.duplicate.append(None)

        return self.changepoints_and_gaps

    def plot_changepoints(self):
        """
        Plot detected changepoints and missing gaps
        """
        if self.fit_predict_called == False:
            raise ValueError('fit_predict() must be called before changepoints can be plotted')
        print(f'Number of changepoints: {len(self.changepoints)}')
        print(f'Number of missing gaps: {len(self.missing_gaps)}')
        plt.figure(figsize=(20,5))
        plt.plot(self.df)
        plt.vlines(self.changepoints, ymin=0, ymax=self.df.max(), color='r', label='changepoint')
        plt.vlines(self.missing_gaps, ymin=0, ymax=self.df.max(), color='c', label='missing gap')
        plt.legend()

    def plot_rolling_window(self, scan_window_start):
        """
        Plot a single rolling window

        Parameters
        ----------
        scan_window_start: pd.Timestamp
            Starting time index of the scan window
        """
        if self.fit_predict_called == False:
            raise ValueError('fit_predict() must be called before changepoint can be plotted')
             
        def pass_or_fail(bool):
            return "Pass" if bool else "Fail"

        scan_window_start = pd.Timestamp(scan_window_start)
        idx = self.scan_window_start_all.index(scan_window_start)
        rolling_window_start = scan_window_start - pd.Timedelta(self.historical_window)
        rolling_window_end = scan_window_start + pd.Timedelta(self.scan_window) + pd.Timedelta(self.buffer_window) - pd.Timedelta(self.freq)
        df = self.df.loc[rolling_window_start:rolling_window_end]

        if self.changepoints_all[idx] == None:
            print('No changepoint detected!')
        else:
            print(f'Changepoint: {self.changepoints_all[idx]}')
            print(f'Missing gap: {self.missing_gap_boolean[idx]}')
            if self.missing_gap_boolean[idx] == False:
                if self.tests["n_day_rule"]:
                    print(f'n_day_rule: {self.n_day_rule_values_all[idx]}')
                if self.n_day_rule_values_all[idx] == 'Pass':
                    print(f'direction: {self.directions_all[idx]}')
                    if self.tests["log_likelihood_ratio_test"]:
                        print(f'llr_7d: {self.llr_7d_pvalues_all[idx]} ({pass_or_fail(self.llr_7d_pvalues_all[idx] < self.llr_threshold)})')
                        print(f'llr_backward: {self.llr_backward_pvalues_all[idx]} ({pass_or_fail(self.llr_backward_pvalues_all[idx] < self.llr_threshold)})')
                        print(f'llr_forward: {self.llr_forward_pvalues_all[idx]} ({pass_or_fail(self.llr_forward_pvalues_all[idx] < self.llr_threshold)})')
                    if self.tests["relative_change_test"]:
                        print(f'relative_change_7d: {self.perc_change_7d_values_all[idx]} ({pass_or_fail(self.perc_change_7d_values_all[idx] >= self.perc_change_threshold)})')
                        print(f'relative_change_backward: {self.perc_change_backward_values_all[idx]} ({pass_or_fail(self.perc_change_backward_values_all[idx] >= self.perc_change_threshold)})')
                        print(f'relative_change_forward: {self.perc_change_forward_values_all[idx]} ({pass_or_fail(self.perc_change_forward_values_all[idx] >= self.perc_change_threshold)})')
                        print(f'relative_change_hist_buffer {self.perc_change_hist_buffer_values_all[idx]} ({pass_or_fail(self.perc_change_hist_buffer_values_all[idx] >= self.perc_change_threshold)})')
                    if self.tests["absolute_change_test"]:
                        print(f'absolute_change_7d: {self.abs_change_7d_values_all[idx]} ({pass_or_fail(self.abs_change_7d_values_all[idx] >= self.abs_change_threshold)})')
                        print(f'absolute_change_backward: {self.abs_change_backward_values_all[idx]} ({pass_or_fail(self.abs_change_backward_values_all[idx] >= self.abs_change_threshold)})')
                        print(f'absolute_change_forward: {self.abs_change_forward_values_all[idx]} ({pass_or_fail(self.abs_change_forward_values_all[idx] >= self.abs_change_threshold)})')
                    if self.tests["magnitude_test"]:
                        print(f'magnitude_test: {self.mag_values_all[idx]} ({pass_or_fail(self.mag_values_all[idx] >= self.magnitude_comparable_day)})')
                    print(f'duplicate: {self.duplicate[idx]}')

        if self.preprocessing_window == 'full':
            pp = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
            df_pp = pp.fit_transform(df)
            outliers = pp.outliers
        else:
            pp_hist = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
            hist_pp = pp_hist.fit_transform(df[:self.historical_window_periods])
            pp_scan = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
            scan_pp = pp_scan.fit_transform(df[self.historical_window_periods:self.historical_window_periods+self.scan_window_periods])
            pp_buffer = Preprocessor(self.freq, self.na_th, self.missing_gap_threshold, self.iqr_coeff, self.ewm, self.alpha)
            buffer_pp = pp_buffer.fit_transform(df[-self.buffer_window_periods:])
            df_pp = hist_pp.append(scan_pp.append(buffer_pp))
            outliers = pp_hist.outliers.append(pp_scan.outliers.append(pp_buffer.outliers))

        fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(20,5))
        axs[0].set_title("Before data preprocessing")
        axs[0].plot(df)
        axs[0].axvline(scan_window_start, c='g', label='scan window')
        axs[0].axvline(self.scan_window_end_all[idx], c='g')
        axs[0].vlines(outliers, ymin=0, ymax=df.max(), color='grey', lw=2, linestyles='dotted', label='outlier')
        if self.changepoints_all[idx] != None:
            axs[0].axvline(self.changepoints_all[idx], c='r', label='changepoint')
        axs[0].legend(loc='upper left')

        axs[1].set_title("After data preprocessing")
        axs[1].plot(df_pp)
        axs[1].axvline(scan_window_start, c='g', label='scan window')
        axs[1].axvline(self.scan_window_end_all[idx], c='g')
        if self.changepoints_all[idx] != None:
            axs[1].axvline(self.changepoints_all[idx], c='r', label='changepoint')
        

    def generate_test_results(self, all_changepoints=False, exclude_missing_gaps=True):
        """
        Generate test results of the detected changepoints.

        Parameters
        ----------
        all_changepoints: bool, default=False
            If True, test results for all changepoints, including those not flagged out, will be shown.

        exclude_missing_gaps: bool, default=True
            If True, test results will exclude changepoints due to missing gaps.

        Returns
        -------
        results: pandas.DataFrame
            Dataframe containing test results of each changepoint
        """
        if self.fit_predict_called == False:
            raise ValueError('fit_predict() needs to be called before test results can be generated')

        results = pd.DataFrame({
            'changepoint': self.changepoints_all,
            'scan_window_start': self.scan_window_start_all,
            'scan_window_end': self.scan_window_end_all,
            'passed_all_tests': self.passed_all_tests,
            'duplicate': self.duplicate,
            'missing_gap': self.missing_gap_boolean,
            'direction': self.directions_all,
            'n_day_rule': self.n_day_rule_values_all,
            'llr_pvalue_7d': self.llr_7d_pvalues_all, 
            'llr_pvalue_backward': self.llr_backward_pvalues_all, 
            'llr_pvalue_forward': self.llr_forward_pvalues_all, 
            'mag_test_value': self.mag_values_all, 
            'perc_change_7d': self.perc_change_7d_values_all, 
            'perc_change_backward': self.perc_change_backward_values_all, 
            'perc_change_forward': self.perc_change_forward_values_all,
            'perc_change_hist_buffer': self.perc_change_hist_buffer_values_all,
            'abs_change_7d': self.abs_change_7d_values_all,
            'abs_change_backward': self.abs_change_backward_values_all, 
            'abs_change_forward': self.abs_change_forward_values_all,
        })
        if all_changepoints:
            return results
        else:
            filtered_results = results[results['passed_all_tests']]
            filtered_results = filtered_results[filtered_results['duplicate']==False]
            missing_gap_results = results[results['missing_gap']]
            if exclude_missing_gaps:
                missing_gap_results = missing_gap_results[missing_gap_results['changepoint'].isin(self.missing_gaps)].drop_duplicates(subset='changepoint')
            filtered_results = filtered_results.append(missing_gap_results)
            return filtered_results.sort_values(by='scan_window_start').reset_index(drop=True)
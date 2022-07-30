# **Change Point Detection**

---

This project aims to:

* Detect changepoints in a univariate time series in a near-online manner

---

## User Guide
### Importing the modules
```
from detector import Detector
```

### Detector
* Conducts change point detection and condition checks on a rolling window basis
* Input: pandas.DataFrame indexed by DateTime (row: datetime, col: observations)
*  `fit_predict(X)`: Detects and returns changepoints

Parameters:

== Preprocessing ==
* `freq`: Resampling frequency of time series. `str, default='30min'`
* `agg`: Aggregation function for resampling. `{'mean', 'median', 'std', 'min', or 'max'), default='mean'`
* `missing_gap_threshold`: Missing gaps >= missing_gap_threshold will be reported. `str, default='1d'`
* `iqr_coeff`: Used to obtain outlier profile of each time index for outlier handling. `float, default=5`
* `ewm`: If True, time series will be exponentially weighted. `bool, default=True`
* `alpha`: Smooting factor for exponential smoothing. `float, default=0.9`
*  `preprocessing_window`: Specify whether historical, scan and buffer window will be preprocessed separately or as a whole. `{'full', 'separate'}, default='full'`

== Change Point Detection ==
* `cpd_algorithm`: Algorithm for changepoint detection. `{'dynp', 'binseg', 'cusum', 'bocpd'}, default='dynp'`
* `cpd_cost`: Cost function for changepoint detection. `str, default='l2'`
* `historical_window`: Number of days in historical window. `str, default=14d`
* `scan_window`: Number of days in scan window. `str, default=7d`
* `buffer_window`: Number of days in buffer window. `str, default=7d`
* `step`: Number of days the window rolls ahead for each iteration. `str, default=1d`

== Condition Checks ==
* `tests`: Specify which tests to be conducted on changepoints. `dict`
* `n_day_rule_threshold`: Minimum number of days between changepoint and scan window boundaries. `str, default=1d`
* `llr_threshold`: Significance level for log-likelihood tests. `float, default=0.01`
* `magnitude_quantile`: Percentile value used in magnitude test. `float, default=0.05`
* `magnitude_quantile_direction`: Direction of percentile value used in magnitude test. `{'top', 'bottom', 'median', 'mean', 'all', 'any'}, default='any'`
* `magnitude_ratio`: Used to generate the comparable ratio for magnitude test. `float, default=0.2`
* `magnitude_ratio_multiplier`: Used to generate the comparable ratio for magnitude test. `float, default=0.75`
* `magnitude_comparable_day`: Minimum proportion of sliding windows for magnitude test. `float, default=0.5`
* `perc_change_threshold`: Threshold for relative change tests. `float, default=0.1`
* `abs_change_threshold`: Threshold for absolute change tests. `float, default=0.1`


### Workflow
1. Initialize a detector. All parameters are listed above and have default values.
```
detector = Detector()
```

2. Call fit_predict() with an input time series dataframe to detect changepoints.
```
changepoints = detector.fit_predict(df)
```

3. Call plot_changepoints() to plot changepoints and missing gaps.
```
detector.plot_changepoints()
```

4. Call plot_rolling_window(scan_window_start) to plot a specific rolling window. Data before and after preprocessing will be plotted. The changepoint, if any, will also be plotted and its test results will be printed.
```
detector.plot_rolling_window(scan_window_start)
```
5. Call generate_test_results() to generate test results for the detected changepoints. Set all_changepoints=True to see test results for all changepoints, including those not flagged out. Set exclude_missing_gaps=False to include changepoints due to missing gap.
```
detector.generate_test_results()
detector.generate_test_results(all_changepoints=True)
detector.generate_test_results(exclude_missing_gaps=False)
```

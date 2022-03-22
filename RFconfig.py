# coding: utf-8

"""
MiDAS - Mission Driven Artificial Lift Intelligent Systems

    MiDAS ESP Replacement Forecast Configuration file

    Last modified March 2022

Permissions:
    This code and its documentation can integrated with company
    applications provided the unmodified code and this notice
    are included.

    This code cannot be copied or modified in whole or part.
"""

# Initial Settings
NUMBER_OF_BUCKETS = 5
DATE_FORMAT = '%m/%d/%Y'
MONTH_DEFAULT = 'December'
NEW_WELL_LIST_FILE = 'new_well_list.csv'

# ARIMA settings
CONFIDENCE_INTERVAL = 0.25
YEARS_TO_FORECAST = 3
YEARS_TO_TEST = 0
ADJUST_K = True
OPTIMIZE_ARIMA = False
DEFAULT_ORDER = (1, 1, 0)
BEST_ARIMA_METHOD = 'MSE'
P_VALUES = range(0, 3)
D_VALUES = range(0, 3)
Q_VALUES = range(0, 2)


'''
MiDAS - Mission Driven Artificial Lift Intelligent Systems

    MiDAS ESP Replacement Forecast process library

    Last modified March 2022

Permissions:
    This code and its documentation can integrated with company
    applications provided the unmodified code and this notice
    are included.

    This code cannot be copied or modified in whole or part.
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lifelines import KaplanMeierFitter, WeibullFitter, LogNormalFitter
import warnings

import RFconfig

warnings.filterwarnings('ignore')
date_format = RFconfig.DATE_FORMAT


def esp_mortality_table(df_esp_db, end_data_year, number_of_buckets, end_data_month, end_data_day):
    '''
    Dataframe field details
    - Tx(t) is the remaining lifetime of a unit aged x on January the first of year t;
        this individual will fail at age x + Tx(t) in year t + Tx(t).
    - qx(t) is the probability that an x-aged unit in calendar year t fails before reaching
        age x + 1, i.e. qx(t) = Pr[Tx(t) ' 1].
    - px(t) = 1−qx(t) is the probability that an x-aged unit in calendar year t reaches
        age x + 1, i.e. px(t) = Pr[Tx(t) > 1].
    - μx(t) is the failure rate at age x during calendar year t.
    - ex(t) = E[Tx(t)] is the expected remaining lifetime of a unit aged x in year t.
    - ETRxt is the exposure-to-risk at age x during year t, i.e. the total time run by units
        aged x in year t.
    - Dxt is the number of failures recorded at age x during year t, from an exposure-to-risk
        ETRxt.
    - Lxt is the number of units aged x on January 1 of year t.
    '''

    start_data_year = pd.to_datetime(df_esp_db['INSTALLATION DATE'], format=date_format).min().year

    # calculate start and end dates for each year in the analysis
    if end_data_month == '12':
        year_off_set = 0
        init_forecast_month = '01'
    else:
        year_off_set = 1  # Adjust year in case a cut off date different from year end
        init_forecast_month = str(int(end_data_month) + 1)
    init_day = '01'
    stop_month = end_data_month
    stop_day = end_data_day
    init_day_month = init_forecast_month + '/' + init_day + '/'
    stop_day_month = stop_month + '/' + stop_day + '/'

    # print(init_day_month, stop_day_month)

    # Create buckets
    max_life = 36500  # maximum for the last open bucket
    t_unit = 365  # time unit for the bucket, default is 365 days
    buckets = [(x * t_unit, (x + 1) * t_unit) for x in range(number_of_buckets-1)]
    buckets.append(((number_of_buckets-1) * t_unit, max_life))  # last bucket

    df_esp_mortality = pd.DataFrame()  # create output dataframe
    year_range = range(start_data_year, end_data_year + 1)

    for year in year_range:  # loop for every year
        esp_exposure_dict = {x: y for x, y in
                             zip(buckets, [0 for y in year_range])}  # empty dictionary of exposures during year
        esp_units_year_end_dict = {x: y for x, y in
                                   zip(buckets, [0 for y in year_range])}  # empty dictionary unit at year end
        date_init = pd.to_datetime(init_day_month + str(year - year_off_set), format=date_format)
        date_stop = pd.to_datetime(stop_day_month + str(year), format=date_format)

        # print(date_init, date_stop)

        operating_condition = (df_esp_db['inst_date'] <= date_stop) & (df_esp_db['fail_date'] >= date_init)
        operating_year_start_condition = (df_esp_db['inst_date'] < date_init) & (df_esp_db['fail_date'] >= date_init)

        df_operating_during_year = df_esp_db[operating_condition]  # pumps operating during the year
        df_operating_year_start = df_esp_db[operating_year_start_condition]  # pumps operating at year start

        cond1 = df_operating_during_year['fail_date'] <= date_stop
        action1 = (df_operating_during_year['fail_date'] - df_operating_during_year['inst_date']).dt.days
        action2 = (date_stop - df_operating_during_year['inst_date']).dt.days
        df_operating_during_year['op_days'] = np.where(cond1, action1, action2)

        df_operating_year_start['op_days'] = (date_init - df_operating_year_start['inst_date']).dt.days

        failed_condition = df_operating_during_year['fail_date'] <= date_stop
        df_operating_failed = df_operating_during_year[failed_condition]  # pumps failed during the year
        df_operating_nonfailed = df_operating_during_year[~failed_condition]  # pumps not failed by year end

        for index, row in df_operating_during_year.iterrows():  # Calculate exposures
            if row.inst_date >= date_init:  # installed during the year
                days_at_year_init = 0
                if row.fail_date <= date_stop:  # failed before year end
                    days_during_year = (row.fail_date - row.inst_date).days
                else:  # failed after year end
                    days_during_year = (date_stop - row.inst_date).days
            else:  # installed prior to year init
                days_at_year_init = (date_init - row.inst_date).days
                if row.fail_date <= date_stop:  # failed before year end
                    days_during_year = (row.fail_date - date_init).days
                else:  # failed after year end
                    days_during_year = (date_stop - date_init).days

            days_at_year_end = days_at_year_init + days_during_year + 1

            for a, b in buckets:
                if (days_at_year_init >= a) & (days_at_year_init < b):  # worked inside
                    esp_exposure_dict[(a, b)] += 1

                if (days_at_year_end >= a) & (days_at_year_end < b) & (
                        days_at_year_init < a):  # worked before and inside
                    esp_exposure_dict[(a, b)] += 1

                if (days_at_year_end >= a) & (days_at_year_end < b) & (row.fail_date > date_stop):  # worked inside
                    esp_units_year_end_dict[(a, b)] += 1

        previous_years_condition = df_esp_db['inst_date'] < date_init
        year_start_well_lst = df_esp_db[previous_years_condition][
            'WELL'].unique()  # list of pumps operating previous years
        year_end_well_lst = df_operating_during_year['WELL'].unique()  # list of pumps operating during the year

        for a, b in buckets:  # Calculate mortality rates.
            cond1 = (df_operating_failed['op_days'] >= a) & \
                    (df_operating_failed['op_days'] < b)
            cond2 = (df_operating_year_start['op_days'] >= a) & \
                    (df_operating_year_start['op_days'] < b)

            df_bucket = df_operating_failed[cond1]
            df_bucket2 = df_operating_year_start[cond2]

            Tx = esp_exposure_dict[(a, b)]
            dx = df_bucket.shape[0]
            u_end = esp_units_year_end_dict[(a, b)]
            u_start = df_bucket2.shape[0]
            if Tx == 0:
                mx = 0
            else:
                mx = dx / Tx

            if (a, b) == buckets[0]:
                number_of_new_wells = len(list(set(year_end_well_lst) - set(year_start_well_lst)))
            else:
                number_of_new_wells = 0
            df_esp_mortality_row = pd.DataFrame({
                'Year': year,
                'Age': str(a) + '-' + str(b),
                'Age2': str(int(a/365)) + '-' + str(int(b/365)) + 'y',
                'Age3': a + 182.5,
                'mx': mx,
                'dx': dx,
                'Tx': Tx,
                'units_year_start': u_start,
                'units_year_end': u_end,
                'new_wells': number_of_new_wells,
            }, index=[0])
            df_esp_mortality = df_esp_mortality.append(df_esp_mortality_row, ignore_index=True)

    df_mtbp = calculate_mtbp(df_esp_db,start_data_year,end_data_year, stop_day_month)

    # Add mtbp to mortality table
    mtbf_columns= pd.DataFrame()
    for i in range(start_data_year, end_data_year + 1):
        mtbf_cols = df_mtbp.loc[df_mtbp.index.repeat(number_of_buckets)].reset_index(drop=True)
        mtbf_columns = mtbf_columns.append(mtbf_cols, ignore_index=True)
    mtbf_columns = mtbf_columns.drop(columns=['Year'])

    df_esp_mortality = df_esp_mortality.join(mtbf_columns)
    # print(df_esp_mortality)

    return df_esp_mortality


def calculate_mtbp(df_esp_db,start_data_year,end_data_year, stop_day_month):
    # Add MTBF Data
    df_life_data = df_esp_db[['inst_date', 'fail_date', 'STATUS']].copy()

    wait_years = 3
    df_mtbp = pd.DataFrame()
    for year in range(start_data_year, end_data_year + 1):  # loop for every year
        if year > start_data_year + wait_years:
            date_stop = pd.to_datetime(stop_day_month + str(year), format=date_format)
            existing_year_end_condition = (df_life_data['inst_date'] <= date_stop)
            df_exist_year_end = df_life_data[existing_year_end_condition]
            df_exist_year_end = df_exist_year_end.assign(datestop=date_stop)

            cond1 = (df_exist_year_end['fail_date'] <= date_stop)
            action1 = (df_exist_year_end['fail_date'] - df_exist_year_end['inst_date']).dt.days + 0.01
            action2 = (df_exist_year_end['datestop'] - df_exist_year_end['inst_date']).dt.days + 0.01
            df_exist_year_end['op_days'] = np.where(cond1, action1, action2)

            cond1 = (df_exist_year_end['fail_date'] <= date_stop)
            df_exist_year_end['censored'] = np.where(cond1, 1, 0)

            # print( df_exist_year_end[df_exist_year_end['op_days']<=0]['op_days'])
            # print(df_exist_year_end)
            durations = df_exist_year_end['op_days']
            event_observed = df_exist_year_end['censored']

            # kmf = KaplanMeierFitter().fit(durations, event_observed)
            lnf = LogNormalFitter().fit(durations, event_observed)
            PR = lnf.hazard_.iloc[-1, 0]
            if PR > 0:
                MTBP = 1 / PR
        else:
            PR = 0
            MTBP = 0

        df_mtbp_row = pd.DataFrame({
            'Year': year,
            'PR': PR,
            'MTBP': MTBP,
        }, index=[0])
        df_mtbp = df_mtbp.append(df_mtbp_row, ignore_index=True)

    return df_mtbp


def process_rul_life_data(df_raw_esp_data, end_data_month_name, end_data_year, number_of_buckets):
    month_days_dict = {'01': '31', '02': '28', '03': '31', '04': '31', '05': '31', '06': '30',
                       '07': '31', '08': '31', '09': '30', '10': '31', '11': '30', '12': '31'}
    month_number_dict = {'January': '01', 'February': '02', 'March': '03', 'April': '04',
                         'May': '05', 'June': '06', 'July': '07', 'August': '08', 'September': '09',
                         'October': '10', 'November': '11', 'December': '12'}

    df_esp_database = df_raw_esp_data.copy()  # make a working copy of the database

    # Format end of data date and begin of forecast day
    end_data_month = month_number_dict[end_data_month_name]
    end_data_day = month_days_dict[end_data_month]
    end_data_date_str = end_data_month + '/' + end_data_day + '/' + str(end_data_year)
    end_data_date = pd.to_datetime(end_data_date_str, format=date_format)
    running_cut_off_date = pd.to_datetime(end_data_date_str, format=date_format) + timedelta(days=1)
    running_cut_off_date_str = running_cut_off_date.strftime(format=date_format)

    # Remove duplicates from the original Oracle File
    df_esp_database.drop_duplicates(subset=['WELL', 'INSTALLATION DATE'], inplace=True)

    # Remove installations newer than end_data_date
    ins_date = pd.to_datetime(df_esp_database['INSTALLATION DATE'], format=date_format)
    df_esp_database = df_esp_database[ins_date <= end_data_date]

    # Remove Failure Dates newer than end_data_date
    fdd = pd.to_datetime(df_esp_database['FAILURE DATE'], format=date_format)
    fd = df_esp_database['FAILURE DATE']
    df_esp_database['FAILURE DATE'] = np.where(fdd > end_data_date, np.nan, fd)

    # Remove Removal Dates newer than end_data_date
    if 'REMOVAL DATE' in df_esp_database.columns:
        rdd = pd.to_datetime(df_esp_database['REMOVAL DATE'], format=date_format)
        rd = df_esp_database['REMOVAL DATE']
        df_esp_database['REMOVAL DATE'] = np.where(rdd > end_data_date, np.nan, rd)

    # Assign failure date to those failures considered 'OTH' in Oracle
    fd = df_esp_database['FAILURE DATE']
    rd = df_esp_database['REMOVAL DATE']
    df_esp_database['STATUS'] = np.where(pd.isnull(fd) & ~pd.isnull(rd), 'NONFAILURE', 'FAILURE')
    df_esp_database['FAILURE DATE'] = np.where(pd.isnull(fd) & ~pd.isnull(rd), rd, fd)

    # Assign status and failure date for currently operating pumps
    fd = df_esp_database['FAILURE DATE']  # update fd
    st = df_esp_database['STATUS']
    df_esp_database['STATUS'] = np.where(pd.isnull(fd), 'RUNNING', st)
    st = df_esp_database['STATUS']  # update st
    df_esp_database['FAILURE DATE'] = np.where(st == 'RUNNING', running_cut_off_date_str, fd)

    # Convert date strings to datetime format
    df_esp_database['fail_date'] = pd.to_datetime(df_esp_database['FAILURE DATE'], format=date_format)
    df_esp_database['Year'] = pd.DatetimeIndex(df_esp_database['fail_date']).year
    df_esp_database['inst_date'] = pd.to_datetime(df_esp_database['INSTALLATION DATE'], format=date_format)
    if 'REMOVAL DATE' in df_esp_database.columns:
        df_esp_database['remov_date'] = pd.to_datetime(df_esp_database['REMOVAL DATE'], format=date_format)
    if 'START DATE' in df_esp_database.columns:
        df_esp_database['start_date'] = pd.to_datetime(df_esp_database['START DATE'], format=date_format)

    # Add run days
    df_esp_database['run_days'] = (df_esp_database['fail_date'] - df_esp_database['inst_date']).dt.days

    # Create mortality table
    df_esp_mortality_table = esp_mortality_table(df_esp_database, end_data_year,
                                                 number_of_buckets,
                                                 end_data_month, end_data_day)

    return df_esp_database, df_esp_mortality_table

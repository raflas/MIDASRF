
"""
MiDAS - Mission Driven Artificial Lift Intelligent Systems

    MiDAS ESP Replacement Forecast - Lee Carter modeling routines

    Last modified March 2022

Permissions:
    This code and its documentation can integrated with company
    applications provided the unmodified code and this notice
    are included.

    This code cannot be copied or modified in whole or part.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

import RFconfig

years_to_forecast = RFconfig.YEARS_TO_FORECAST
years_to_test = RFconfig.YEARS_TO_TEST
number_of_buckets = RFconfig.NUMBER_OF_BUCKETS
confidence_interval = RFconfig.CONFIDENCE_INTERVAL,
adjust_k = RFconfig.ADJUST_K
optimize_arima = RFconfig.OPTIMIZE_ARIMA

def find_arima_best_order(dataset):
    # Finds the best ARIMA p,d,q combination to fit the data.

    # p, d, q test ranges for optimization
    p_values = RFconfig.P_VALUES
    d_values = RFconfig.D_VALUES
    q_values = RFconfig.Q_VALUES

    def evaluate_arima_model(X, arima_order):  # evaluate an ARIMA model for a given order (p,d,q)
        # calculates mse, aic, bic for a fit with given p,d,q order
        # prepare training dataset
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit_try = model.fit()
            yhat = model_fit_try.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        mse = mean_squared_error(test, predictions)
        aic = model_fit_try.aic
        bic = model_fit_try.bic
        return mse, aic, bic

    dataset = dataset.astype('float32')
    best_mse, best_msecfg = float('inf'), None
    best_aic, best_aiccfg = float('inf'), None
    best_bic, best_biccfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse, aic, bic = evaluate_arima_model(dataset, order)
                    if mse < best_mse:
                        best_mse, best_msecfg = mse, order
                    if aic < best_aic:
                        best_aic, best_aiccfg = aic, order
                    if bic < best_bic:
                        best_bic, best_biccfg = bic, order
                except:
                    continue
    return best_msecfg, best_aiccfg, best_biccfg


def mx_calculation(a, b, k, years_to_model, number_of_buckets):
    # Lee Carter's mxhat calculation with provided a,b,k
    Mhat = np.zeros((years_to_model, number_of_buckets))
    for i in range(len(k)):
        mxhat = np.exp(a + k[i] * b)
        Mhat[i, :] = mxhat.transpose()
    return Mhat


def model_fit_evaluation(espmx, Mhat, min_year, max_year, number_of_buckets):
    # Estimates yearly failures using model's Mhat
    espmx_aug = pd.DataFrame()
    df_dx = pd.DataFrame()

    zv = np.repeat(0., number_of_buckets)
    one = np.repeat(1., number_of_buckets)

    for year in range(min_year, max_year + 1):
        esp_year = espmx[espmx['Year'] == year]
        esp_year.reset_index(inplace=True, drop=True)

        esp_year['mxhat'] = Mhat[year - min_year].transpose()
        Utxf = esp_year.mxhat.to_numpy()
        St = esp_year.units_year_start.to_numpy()
        Newt = esp_year.new_wells.to_numpy()  # New units on that year, go to age group 0
        # Newt = np.concatenate([[new_wells_df[year]],zv[1:]]) # New units on that year, go to age group 0

        Fsum = np.concatenate(
            [[np.sum(St * Utxf)], zv[1:]])  # Sum of units failed from St and where replaced, go to age group 0
        Surv = St - St * Utxf
        sroll = np.roll(Surv, 1)  # unit survived rolled

        # Units that survived from St and rolled to next group
        Surv0 = np.concatenate([zv[:1], sroll[1:]])  # # remove first element [0,sroll[1:]]
        SurvR = Surv0 + np.concatenate([zv[:-1], [Surv[-1]]])  # # add last [0,0,0,...,Surv[-1]]

        esp_year['carry_over'] = Surv0
        esp_year['survived'] = Surv
        esp_year['Txhat'] = St + Surv0 + (Newt + Fsum) * (one + Utxf)
        esp_year['dxhat'] = esp_year['Txhat'] * Utxf

        df_dx_row = pd.DataFrame({
            'Year': year,
            'Dx': esp_year.dx.sum(),
            'Dxhat': esp_year.dxhat.sum(),
        }, index=[0])

        df_dx = df_dx.append(df_dx_row, ignore_index=True)
        espmx_aug = espmx_aug.append(esp_year, ignore_index=True)

    # print('MAE=',mean_absolute_error(df_dx.Dx, df_dx.Dxhat))
    # print('MSE=',mean_squared_error(df_dx.Dx, df_dx.Dxhat))

    return espmx_aug, df_dx


def ESP_replacement_calculation(espmx_aug, Mhat, start_year, end_year, new_wells_df, scenario_name):
    # Forecasts yearly failures using model's Mhat
    df_forecast = espmx_aug.copy()

    zv = np.repeat(0., number_of_buckets)
    one = np.repeat(1., number_of_buckets)

    for year in range(start_year, end_year + 1):
        new_wells_year =  new_wells_df[new_wells_df['Year']==year]['NewWells'].values[0]
        forecast_year = df_forecast[df_forecast['Year'] == year - 1]
        forecast_year.reset_index(inplace=True, drop=True)
        forecast_year.Year = [year for x in range(forecast_year.shape[0])]
        forecast_year.mx = [0 for x in range(forecast_year.shape[0])]
        forecast_year.dx = [0 for x in range(forecast_year.shape[0])]
        forecast_year.Tx = [0 for x in range(forecast_year.shape[0])]
        forecast_year['mxhat'] = Mhat[year - start_year].transpose()
        forecast_year.loc[0, 'new_wells'] = new_wells_year
        forecast_year['units_year_start'] = forecast_year['units_year_end']
        forecast_year['scenario'] = [scenario_name for x in range(forecast_year.shape[0])]

        Utxf = forecast_year.mxhat.to_numpy()

        St = forecast_year['units_year_start'].to_numpy()
        Newt = np.concatenate([[new_wells_year], zv[1:]])  # New units on that year, go to age group 0
        Fsum = np.concatenate(
            [[np.sum(St * Utxf)], zv[1:]])  # Sum of units failed from St and where replaced, go to age group 0

        Surv = St - St * Utxf

        sroll = np.roll(Surv, 1)  # unit survived rolled

        # Units that survived from St and rolled to next group
        Surv0 = np.concatenate([zv[:1], sroll[1:]])  # # remove first element [0,sroll[1:]]
        SurvR = Surv0 + np.concatenate([zv[:-1], [Surv[-1]]])  # # add last [0,0,0,...,Surv[-1]]

        forecast_year['carry_over'] = Surv0
        forecast_year['survived'] = Surv
        forecast_year['Txhat'] = St + Surv0 + (Newt + Fsum) * (one + Utxf)
        forecast_year['dxhat'] = forecast_year['Txhat'] * Utxf

        if year < end_year:
            forecast_year['units_year_end'] = SurvR + Newt + Fsum

        df_forecast = df_forecast.append(forecast_year, ignore_index=True)

    return df_forecast


def lee_carter_model(mx_data, years_to_model, min_year, max_year):
    """
    Obtain parameters for the Lee Carter model
    The vector 'a' can be interpreted as an average mortality for an age group
    The vector 'k' tracks mortality changes over time
    The vector 'b' determines how much each age group changes when k for the year
    """
    # Calculate log of failure rates
    rates = mx_data.mx

    logrates = np.log(rates)
    M = logrates.to_numpy().reshape(years_to_model, number_of_buckets)

    # Calculate
    a = M.mean(axis=0)
    A = np.zeros((years_to_model, number_of_buckets))
    for j in range(number_of_buckets):
        A[:, j] = M[:, j] - a[j]
    U, d, V = np.linalg.svd(A)
    Vt = V.transpose()
    b = Vt[:, 0] / sum(Vt[:, 0])
    k = U[:, 0] * sum(Vt[:, 0]) * d[0]

    trend = pd.DataFrame()
    trend['k'] = k
    year_list = [x for x in range(min_year, max_year+1)]
    trend.index = pd.to_datetime(year_list, format='%Y')
    trend.index = pd.DatetimeIndex(trend.index.values, freq=trend.index.inferred_freq)


    return a, b, k, trend


def run_arima_model(model_order, k, ci, years_to_forecast, verbose=False):
    # Uses ARIMA to create a model for kt and forecast for years_to_forecast
    model = ARIMA(k, order=model_order)  # define model
    model_fit_ = model.fit()  # Fit model

    forecast = model_fit_.get_forecast(years_to_forecast)
    df_predictions = forecast.summary_frame(alpha=RFconfig.CONFIDENCE_INTERVAL)

    if df_predictions.isnull().values.any():
        print('Model did not converge')

    if verbose:
        # summary of fit model
        print('Model order ', model_order)
        print(model_fit_.summary(alpha=ci))
        print('Forecast Summary')
        print(forecast.summary_frame(alpha=ci))

    df_k = pd.DataFrame()
    df_k_forecast = pd.DataFrame()
    df_k['Year'] = k.index.year
    df_k['k'] = k.to_list()
    df_k['khat'] = k.to_list()
    df_k_forecast['Year'] = df_predictions.index.year
    df_k_forecast['khat'] = df_predictions['mean'].to_list()
    df_k_forecast['khat_ci_lower'] = df_predictions['mean_ci_lower'].to_list()
    df_k_forecast['khat_ci_upper'] = df_predictions['mean_ci_upper'].to_list()
    df_k_forecast = df_k.append(df_k_forecast, ignore_index=True)

    return df_k_forecast


def k_adjustment(a, b, k, espmx, years_to_model, min_year, max_year, max_iterations=100, epsilon=3):
    # Fine tunes Kt to fit the actual failure data
    k_values = k.copy()
    max_value = max(abs(k_values.max()), abs(k_values.min()))
    k_inc = max_value / 50
    i = 0
    convergence = False

    old_diff = [0 for x in range(min_year, max_year + 1)]
    direction = [1 for x in range(min_year, max_year + 1)]
    espmx_aug_ = pd.DataFrame()
    df_dx_ = pd.DataFrame()
    while (i < max_iterations) & (not convergence):
        Mhat = mx_calculation(a, b, k_values, years_to_model, number_of_buckets)
        espmx_aug_, df_dx_ = model_fit_evaluation(espmx, Mhat, min_year, max_year, number_of_buckets)
        convergence = True
        for index, row in df_dx_.iterrows():
            year = row.Year
            diff = row.Dx - row.Dxhat
            j = int(year - min_year)
            if abs(diff) > epsilon:
                convergence = False
                if abs(old_diff[j]) < abs(diff):
                    direction[j] = direction[j] * -1
                k_values[j] += direction[j] * k_inc
            old_diff[j] = diff
        i += 1
    return k_values, espmx_aug_, df_dx_


def esp_replacement_scenarios(esp_mx_forecast, df_k_forecast, a, b, new_wells_df):
    """
    Calculate three ESP replacement scenarions:
        - for the mean
        - for the lower confidence interval
        - for the upper confidence interval
    """

    df_khat = df_k_forecast[['Year', 'khat', 'khat_ci_lower', 'khat_ci_upper']].dropna()

    start_year = df_khat.Year.min()
    end_year = df_khat.Year.max()

    # get predicted k
    khat_mean = df_khat['khat'].to_list()
    khat_ci_lower = df_khat['khat_ci_lower'].to_list()
    khat_ci_upper = df_khat['khat_ci_upper'].to_list()

    # calculate predicted mx
    Mhat_mean = mx_calculation(a, b, khat_mean, years_to_forecast, number_of_buckets)
    Mhat_ci_lower = mx_calculation(a, b, khat_ci_lower, years_to_forecast, number_of_buckets)
    Mhat_ci_upper = mx_calculation(a, b, khat_ci_upper, years_to_forecast, number_of_buckets)

    # calculate ESP replacements
    espmx_ci_lower = ESP_replacement_calculation(esp_mx_forecast, Mhat_ci_lower, start_year, end_year,
                                                 new_wells_df, 'ci_lower')
    espmx_mean = ESP_replacement_calculation(esp_mx_forecast, Mhat_mean, start_year, end_year, new_wells_df,
                                             'mean')
    espmx_ci_upper = ESP_replacement_calculation(esp_mx_forecast, Mhat_ci_upper, start_year, end_year,
                                                 new_wells_df,'ci_upper')

    df_scenarios = esp_mx_forecast[esp_mx_forecast['Year'] < start_year]
    df_scenarios = df_scenarios.assign(scenario =['current' for x in range(df_scenarios.shape[0])])
    df_scenarios = df_scenarios.append(espmx_ci_lower[espmx_ci_lower['Year'] >=start_year], ignore_index=True)
    df_scenarios = df_scenarios.append(espmx_mean[espmx_mean['Year'] >=start_year], ignore_index=True)
    df_scenarios = df_scenarios.append(espmx_ci_upper[espmx_ci_upper['Year'] >=start_year], ignore_index=True)

    df_k_forecast2 = df_k_forecast[df_k_forecast['Year'] < start_year]
    df_k_forecast2['khat_ci_lower'] = df_k_forecast2['khat']  # to avoid nan cells
    df_k_forecast2['khat_ci_upper'] = df_k_forecast2['khat']  # to avoid nan cells

    k_columns = df_k_forecast2.loc[df_k_forecast2.index.repeat(number_of_buckets)].reset_index(drop=True)
    for i in range (years_to_forecast):
        k_columns2 = df_khat.loc[df_khat.index.repeat(number_of_buckets)].reset_index(drop=True)
        k_columns = k_columns.append(k_columns2, ignore_index=True)
    k_columns = k_columns.drop(columns=['Year', 'k'])

    df_scenarios = df_scenarios.join(k_columns)
    df_scenarios = df_scenarios.assign(a= np.tile(a, int(df_scenarios.shape[0]/number_of_buckets)))
    df_scenarios = df_scenarios.assign(b= np.tile(b, int(df_scenarios.shape[0]/number_of_buckets)))


    return  df_scenarios


def run_forecasts(mx_data_filtered, min_year, max_year, new_wells_df, years_to_model, df_mx_test_data):
    # Sort database and make a copy
    filtered_mx_data = mx_data_filtered.sort_values(['Year', 'Age3'], ascending=True).copy()

    # Run Lee Carter model
    a, b, k, trend = lee_carter_model(filtered_mx_data, years_to_model, min_year, max_year)

    # Adjust k
    if adjust_k:
        k_adj, mx_forecast, dx_forecast = k_adjustment(a, b, k, filtered_mx_data, years_to_model,
                                               min_year, max_year, max_iterations=100,epsilon=3)
        trend['k_values'] = k_adj
    else:
        Mhat = mx_calculation(a, b, k, years_to_model, number_of_buckets)
        mx_forecast, dx_forecast = model_fit_evaluation(filtered_mx_data, Mhat, min_year, max_year, number_of_buckets)
        trend['k_values'] = k

    # Get best order parameters for ARIMA model
    if optimize_arima:
        best_mse_order, best_aic_order, best_bic_order = find_arima_best_order(trend.k_values)
        best_order = {'MSE' : best_mse_order, 'AIC' : best_aic_order, 'BIC' : best_bic_order }
        order = best_order[RFconfig.BEST_ARIMA_METHOD]
    else:
        order = RFconfig.DEFAULT_ORDER

    # Forecast k using ARIMA
    k_forecast = run_arima_model(order, trend.k_values, confidence_interval, years_to_forecast, verbose=False)

    # Generate the replacement forecast's scenarios
    df_scenarios = esp_replacement_scenarios(mx_forecast, k_forecast, a, b, new_wells_df)
    # print(a, b, k, k_adj)

    return df_scenarios




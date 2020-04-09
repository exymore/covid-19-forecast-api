import pandas as pd
import numpy as np
from fbprophet import Prophet
import pickle
import math
import scipy.optimize as optim
from datetime import datetime, timedelta
import requests
from celery import shared_task
import logging
import os



logging.getLogger('fbprophet').setLevel(logging.WARNING)


def build_country_data(country):
    res = []
    keys = country.get('timeline').get('cases').keys()
    for key in keys:
        target_entry = {}
        target_entry['Report_Date'] = key
        country_name = country.get('country')
        if country.get('province') != None:
            country_name = country_name + '_' + country.get('province')
        target_entry[country_name + '_cases'] = country.get('timeline').get('cases').get(key)
        target_entry[country_name + '_deaths'] = country.get('timeline').get('deaths').get(key)
        target_entry[country_name + '_recovered'] = country.get('timeline').get('recovered').get(key)
        res.append(target_entry)
    return res


def build_covid19_data():
    request_str = 'https://corona.lmao.ninja/v2/historical?lastdays=all'
    response = requests.get(request_str)
    json_data = response.json() if response and response.status_code == 200 else None

    df = None
    for country in json_data:
        res = build_country_data(country)
        if df is None:
            df = pd.DataFrame(res)
            df.index = pd.DatetimeIndex(df['Report_Date'])
            df = df.drop('Report_Date', 1)
            df = df.sort_values(by=['Report_Date'])
        else:
            df_new = pd.DataFrame(res)
            df_new.index = pd.DatetimeIndex(df_new['Report_Date'])
            df_new = df_new.drop('Report_Date', 1)
            df_new = df_new.sort_values(by=['Report_Date'])
            df = df.merge(df_new, left_index=True, right_index=True)

    df.to_csv('covid_19_app/forecast/data/covid19_data.csv')
    print("Data fetched")
    return df


def fetch_data():
    build_covid19_data()


def func_logistic(t, a, b, c):
    return c / (1 + a * np.exp(-b * t))


def detect_growth():
    countries_processed = 0
    countries_stabilized = 0
    countries_increasing = 0

    countries_list = []

    df = pd.read_csv('covid_19_app/forecast/data/covid19_data.csv', parse_dates=True)
    columns = df.columns.values
    for column in columns:
        if column.endswith('_cases'):
            data = pd.DataFrame(df[column].values)

            data = data.reset_index(drop=False)
            data.columns = ['Timestep', 'Total Cases']

            p0 = np.random.exponential(size=3)

            bounds = (0, [100000., 1000., 1000000000.])

            x = np.array(data['Timestep']) + 1
            y = np.array(data['Total Cases'])

            try:
                (a, b, c), cov = optim.curve_fit(func_logistic, x, y, bounds=bounds, p0=p0, maxfev=1000000)

                t_fastest = np.log(a) / b
                i_fastest = func_logistic(t_fastest, a, b, c)

                res_df = df[['Report_Date', column]].copy()
                res_df['fastest_grow_day'] = t_fastest
                res_df['fastest_grow_value'] = i_fastest
                res_df['growth_stabilized'] = t_fastest <= x[-1]
                res_df['timestep'] = x
                res_df['res_func_logistic'] = func_logistic(x, a, b, c)

                if t_fastest <= x[-1]:
                    print('Growth stabilized:', column, '| Fastest grow day:', t_fastest, '| Infections:', i_fastest)
                    res_df['cap'] = func_logistic(x[-1] + 10, a, b, c)
                    countries_stabilized += 1
                else:
                    print('Growth increasing:', column, '| Fastest grow day:', t_fastest, '| Infections:', i_fastest)
                    res_df['cap'] = func_logistic(i_fastest + 10, a, b, c)
                    countries_increasing += 1

                countries_processed += 1
                countries_list.append(column)

                res_df.to_csv('covid_19_app/forecast/data/covid19_processed_data_' + column + '.csv')
            except RuntimeError:
                print('No fit found for: ', column)

    d = {'countries_processed': [countries_processed], 'countries_stabilized': [countries_stabilized],
         'countries_increasing': [countries_increasing]}
    df_c = pd.DataFrame(data=d)
    df_c.to_csv('covid_19_app/forecast/data/covid19_stats_countries.csv')

    df_countries = pd.DataFrame(countries_list)
    df_countries.to_csv('covid_19_app/forecast/data/covid19_countries_list.csv')


def build_model(country):
    df = pd.read_csv('covid_19_app/forecast/data/covid19_processed_data_' + country + '.csv', parse_dates=True)
    df_ = df.copy()
    df = df[['Report_Date', country, 'cap']].dropna()

    df.columns = ['ds', 'y', 'cap']

    m = Prophet(growth="logistic")
    m.fit(df)

    future = m.make_future_dataframe(periods=20)
    future['cap'] = df['cap'].iloc[0]

    forecast = m.predict(future)

    res_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds').y).reset_index()
    res_df['current_date'] = df['ds'].iloc[-1]
    res_df['fastest_growth_day'] = df_['fastest_grow_day'].iloc[-1]
    res_df['growth_stabilized'] = df_['growth_stabilized'].iloc[-1]
    res_df['current_day'] = df_['timestep'].iloc[-1]
    res_df['cap'] = df['cap'].iloc[0]

    res_df.to_csv('covid_19_app/forecast/data/covid19_forecast_data_' + country + '.csv')

    print('Processed:', country)


def calculate_forecast():
    df = pd.read_csv('covid_19_app/forecast/data/covid19_data.csv', parse_dates=True)
    columns = df.columns.values
    for column in columns:
        if column.endswith('_cases'):
            build_model(column)
    print('Forecast calculation completed')


@shared_task
def run_training():
    print("Training started")
    print(os.getcwd())
    fetch_data()
    detect_growth()
    calculate_forecast()
    print("Training ended")

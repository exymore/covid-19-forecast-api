import requests
from datetime import datetime, timedelta
import pandas as pd

def build_country_data(country):
    res = []
    keys = country.get('timeline').get('cases').keys()
    # new_date_key = datetime.strftime(datetime.strptime([*keys][-1], '%m/%d/%y') + timedelta(days=1), "%m/%d/%y")
    for key in keys:
        target_entry = {'Report_Date': key}
        country_name = country.get('country')
        if country.get('province') is not None:
            country_name = country_name + '_' + country.get('province')
        target_entry[country_name + '_cases'] = country.get('timeline').get('cases').get(key)
        target_entry[country_name + '_deaths'] = country.get('timeline').get('deaths').get(key)
        target_entry[country_name + '_recovered'] = country.get('timeline').get('recovered').get(key)
        res.append(target_entry)
    # if country.get('province') is None:
    #     country_name = country.get('country')
    #     target_entry = {'Report_Date': new_date_key, country_name + '_cases': actual.get('cases'),
    #                     country_name + '_deaths': actual.get('deaths'),
    #                     country_name + '_recovered': actual.get('recovered')}
    #     res.append(target_entry)
    return res


def build_covid19_data():
    request_str = 'https://corona.lmao.ninja/v2/historical?lastdays=all'
    response = requests.get(request_str)
    json_data = response.json() if response and response.status_code == 200 else None

    # per_countries_response = requests.get('https://corona.lmao.ninja/countries/')
    # per_countries_actual = per_countries_response.json()
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
            df = df.merge(df_new, how='left', left_index=True, right_index=True)

    df.to_csv('data/covid19_data.csv')
    print("Fetching finished")
    return df

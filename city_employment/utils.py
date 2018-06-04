# encoding=utf-8
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
import unicodedata
import datetime
from copy import deepcopy
from pathos import multiprocessing as mp
import datetime
import MySQLdb
import time
import unicodedata


def data_definition(csv_name):
    """
    Function to read a csv file
    :param csv_name: Name of csv
    :return: Read csv
    """
    if csv_name == '' or csv_name is None:
        return None
    data = pd.read_csv(csv_name, encoding='latin1')
    return data


def text_clean(text):
    """
    Function dedicated to clean the text. Hate spaces, slashes, peso signs or accents.
    :param text: Text
    :return: Whipped out text
    """
    text = ''.join((c for c in unicodedata.normalize('NFD', unicode(text)) if
                    unicodedata.category(c) != 'Mn'))
    return text.strip().replace(' ', '_').replace('/', '_').lower()


def month_name(month_nb):
    months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio',
              'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

    return months[month_nb]


def individual_try(city, data):
    tmp = data.loc[data.municipio == city].sort_values(by='fecha').T

    if 'sector' not in tmp.index:
        new_df = pd.DataFrame(
                              index=[city],
                              columns=tmp.loc['fecha'])
        new_df.loc[city] = tmp.loc['TA'].values
        new_df.columns.names = [None]
    else:
        sectors = tmp.T.sector.unique().tolist()
        tmp = tmp.T
        new_df = pd.DataFrame(data=None,
                              # index=range(len(sectors)),
                              columns=['municipio', 'sector'] +
                                      tmp.fecha.unique().tolist())
        for sector in sectors:
            sector_df = tmp.loc[tmp.sector == sector]
            list_of_values = [city, sector] + list(sector_df.TA.values)
            df_to_add = pd.DataFrame(data=None,
                                     index=[sector],
                                     columns=['municipio', 'sector'] +
                                             sector_df.fecha.unique().tolist())
            df_to_add.loc[sector] = list_of_values
            new_df = pd.concat([new_df, df_to_add])
    return new_df


def order_iieg_db(data):
    data.rename(columns={'division_economica': 'sector'}, inplace=True)
    index = data.loc[data['fecha'].str.contains('28|29|30|31'), 'fecha'].index.values.tolist()
    fecha = data.loc[index, 'fecha'].values.tolist()

    dates_to_replace = map(lambda x: x.replace('31', '01'),
                           map(lambda x: x.replace('30', '01'),
                               map(lambda x: x.replace('29', '01'),
                                   map(lambda x: x.replace('28', '01'), fecha))))
    data.loc[index, 'fecha'] = dates_to_replace

    data.fecha = map(lambda date: datetime.datetime.strptime(date, '%d/%m/%Y'), data.fecha)
    data['municipio'] = map(text_clean,
                            map(lambda x: x.decode('utf8'),
                                data.municipio.tolist()))
    if 'sector' in data.columns:
        data['sector'] = map(text_clean,
                                map(lambda x: x.decode('utf8'),
                                    data.sector.tolist()))
    # import time
    # tic = time.time()
    pool = mp.ProcessingPool(mp.cpu_count())
    results = pool.map(individual_try,
                       data['municipio'].unique().tolist(),
                       [data for i in range(len(data.municipio.unique().tolist()))])

    new_df = pd.concat(results)

    not_first_columns = filter(None, map(lambda x: x if type(x) is not str and x.day != 1 else None,
                                         new_df.columns))
    if len(not_first_columns) > 0:
        new_df.drop(not_first_columns, axis=1, inplace=True)
    new_df.columns = map(lambda x: x if type(x) == str else str(x.year) + '/' + month_name(x.month - 1),
                         new_df.columns)
    time_columns = list(new_df.columns)

    if 'sector' not in data.columns:
        new_df['municipio'] = new_df.index.tolist()
        new_df.drop_duplicates(subset=['municipio'], inplace=True)
        region_city_dict = {region: data.loc[data.region == region, 'municipio'].unique().tolist()
                            for region in data.region.unique()}

        regions_list = [region for i in new_df.index
                        for region in region_city_dict.keys()
                        if new_df.loc[i, 'municipio'] in region_city_dict.get(region)]

        new_df['region'] = regions_list
        sorted_columns = ['region', 'municipio'] + time_columns

        new_df = new_df[sorted_columns]
    else:
        new_df.drop_duplicates(subset=['municipio', 'sector'], inplace=True)
    new_df.index = range(len(new_df))

    return new_df


def get_data(table):
    db = MySQLdb.connect(host='127.0.0.1',
                         user='root',
                         passwd='loc4lpas5',
                         db='iieg_database')
    data = pd.read_sql('SELECT * FROM ' + str(table), db)

    return data


def data_clean(data, sectores=False):
    """
    Function to make a basic cleaning of data.
    :param data: Data to clean
    :param sectores: Default False. If true, names of columns come different
    :return: Cleaned dataframe
    """
    if data is None:
        return None

    # Columns names
    data.columns = map(text_clean, data.columns)

    if sectores:
        special_columns = ['municipio', 'sector']
        data.columns.values[0:2] = special_columns
    else:
        special_columns = ['region', 'municipio']
        data.columns.values[0:2] = special_columns
        sum_index = [index for index in data.index if
                     data.loc[index, 'municipio'] == data.loc[index, 'region']]
        data.drop(sum_index, inplace=True)

    # region and municipio text clean
    for column in special_columns:
        data[column] = map(text_clean,
                           map(lambda x: x.decode('utf8'),
                               data[column]))

    # Dropping nan columns just because I can
    data.drop(data.columns[data.isnull().sum() == len(data)].values, axis=1, inplace=True)
    if 'periodo' in data.columns:
        data.drop(['periodo'], axis=1, inplace=True)

    return data


def sectores(data, city):
    """
    Function to filter sectors by city of interest
    :param data: Cleaned data
    :param city: City of interest
    :return: Filtered sectors
    """
    if data is None:
        return None
    sectors = data.loc[data.municipio == city, 'sector']
    temp_frame = data.iloc[data.loc[data.municipio == city].index]

    df = temp_frame.iloc[:, 2::]
    df.index = sectors

    return df


def lag_definition(data, city, default_lags=24):

    if city is None or data is None:
        return None

    def fac(series, lags=default_lags, qstat=True):
        return acf(filter(None, series), nlags=lags, qstat=qstat)

    def fap(series, lags=default_lags, alpha=0.05):
        return pacf(series, nlags=lags, alpha=alpha)

    series = data.loc[data.municipio == city].values[0][2::]

    limit = 1.96 / len(series) ** 0.5
    limit = {'positive': limit, 'negative': -limit}

    fac_lags, qstat, pvalues = fac(series)
    fac_significant = [pos for pos in range(len(fac_lags)) if fac_lags[pos] > limit.get('positive') or
                       fac_lags[pos] < limit.get('negative')]
    if min(fac_significant) == 0:
        fac_significant.remove(0)

    # lags_acf = [lag for lag in nb_pvalues_h0 if fac_lags[lag] > 0.80]
    fap_lags, confint = fap(series)
    fap_significant = [pos for pos in range(len(fac_lags)) if fap_lags[pos] > limit.get('positive') or
                       fap_lags[pos] < limit.get('negative')]
    if min(fap_significant) == 0:
        fap_significant.remove(0)
    # lags_fap = [val for val, fal in enumerate(fap_lags) if np.abs(fal) >= 0.60 and val > 0]

    max_len_lags = 'fac' if len(fac_significant) > len(fap_significant) else 'fap'
    if max_len_lags == 'fac':
        not_equal_lags = list(set(fap_significant) - set(fac_significant))
        final_lags = fac_significant
    else:
        not_equal_lags = list(set(fac_significant) - set(fap_significant))
        final_lags = fap_significant

    if len(not_equal_lags) > 0:
        final_lags.append(not_equal_lags)

    final_lags = sorted(final_lags)
    # final_lags = lags_fap
    # if len(lags_fap) <= 1:
    #     final_lags = lags_acf

    return final_lags


def transform_spanish_months(month):
    months_spanish = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
                      'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    months_english = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                      'august', 'september', 'october', 'november', 'december']
    months_dict = dict(zip(months_spanish, months_english))
    month = month.split('_')[1]
    return months_dict.get(month)


def dataframe_model(data, city, final_lags, sectors_df, min_date=datetime.date(2000, 1, 1)):
    if data is None or final_lags is None:
        return None

    city_index = data.loc[data.municipio == city].index[0]
    # min_month_row = min_date + datetime.timedelta(days=30 * max(final_lags))
    column_to_begin = 2 + max(final_lags)  # min_month_row.month

    df = pd.DataFrame(index=[data.columns[column_to_begin::]],
                      columns=['t-' + str(number + 1) for number in range(max(final_lags))])

    region_interest_indexes = data.loc[data.region == data.loc[city_index, 'region']].index
    period_std = []
    period_mean = []
    region_sums = []
    for index in range(len(df.index)):
        df.iloc[index, :] = data.loc[city_index][index + 2:index + max(final_lags) + 2].values[::-1]
        period_std.append(np.std(df.iloc[index, :]))
        period_mean.append(np.mean(df.iloc[index, :]))

        # Establishment of region total
        region_totals = 0
        for index_region in region_interest_indexes:
            if index_region != city_index:
                region_data = data.loc[index_region][index + 2:index + max(final_lags) + 2].values[::-1]
                region_totals += np.sum(region_data)

        region_sums.append(region_totals)

    df['response'] = data.loc[city_index][df.index.values]
    df['period_std'] = period_std
    df['period_mean'] = period_mean
    df['region_sums'] = region_sums

    for date in df.index:
        for sector in sectors_df.index:
            prev_month = np.where(sectors_df.columns == date)[0] - 1  # we can't know the month value
            df.loc[date, str(sector)] = sectors_df.loc[sector, sectors_df.columns[prev_month]].values[0]

    for column_object in df.select_dtypes(include=['object']).columns:
        df[column_object] = df[column_object].apply(float)

    df.dropna(axis=1, inplace=True, how='all')
    if 'na' in df.columns:
        df.drop('na', axis=1, inplace=1)

    months = map(transform_spanish_months, df.index)
    month_number = map(lambda month: time.strptime(month, '%B').tm_mon, months)
    df['month'] = month_number
    df.fillna(-999, inplace=True)
    return df


def predict_data(data, city, final_lags, sectors_df):
    """
    Fun to order data in order to have predictable data.
    :return: DataFrame with data to predict
    """
    if data is None or final_lags is None:
        return None

    df = pd.DataFrame(index=[0],
                      columns=['t-' + str(number + 1) for number in range(max(final_lags))])

    df.loc[0] = data.loc[data.municipio == city].values[0][-max(final_lags):][::-1]
    df['period_std'] = np.std(df.loc[0].values)
    df['period_mean'] = np.mean(df.loc[0].values)

    city_index = data.loc[data.municipio == city].index[0]
    region_interest_indexes = list(data.loc[data.region == data.loc[city_index, 'region']].index)
    region_interest_indexes.remove(city_index)

    df['region_sums'] = data.loc[region_interest_indexes].iloc[:, -max(final_lags)::].sum().sum()
    # sectors = sectors_df.iloc[:, -max(final_lags)::].sum(axis=1)
    sectors = sectors_df.iloc[:, sectors_df.shape[1]-1]

    for sector_column in sectors.index:
        df[sector_column] = sectors[sector_column]

    for column_object in df.select_dtypes(include=['object']).columns:
        df[column_object] = df[column_object].apply(float)

    df['month'] = time.strptime(transform_spanish_months(data.columns[len(data.columns) - 1]),
                                '%B').tm_mon + 1
    df.fillna(-999, inplace=True)
    df.dropna(axis=1, inplace=True, how='all')
    if 'na' in df.columns:
        df.drop('na', axis=1, inplace=1)
    return df



import individual_models as im
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from copy import deepcopy
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.pyplot as plt
import argparse
import yaml
import json
import time
import datetime
rcParams.update({'figure.autolayout': True})
plt.style.use('ggplot')


def data_preparation(config_name, yearly, city_name):
    """
    Function to prepare data to use in a city model
    :return: pandas.DataFrame to model
    """
    # Reading basic info
    with open(config_name, "r") as f:
        config = yaml.load(f)
        city_table = config['CITIES_DATA']
        sector_table = config['SECTORS_DATA']
        # city_name = config['CITY']

    # Reading and cleaning data
    tic = time.time()
    city_data = r.data_clean(data=r.order_iieg_db(r.get_data(table=city_table)))
    toc = time.time()
    print toc - tic
    sectors_clean = r.data_clean(data=r.order_iieg_db(r.get_data(table=sector_table)), sectores=True)
    print time.time() - toc

    if yearly:
        first_columns = [['region', 'municipio']]
        columns_ready = [column for column in city_data.columns if 'diciembre' in column]
        first_columns.append(columns_ready)
        columns_ready = [value for sublist in first_columns for value in sublist]

        city_data = city_data[columns_ready]

    # Re-arranging sectors file
    city_sectors = r.sectores(data=sectors_clean,
                              city=city_name)

    # Getting best lags to use for the model
    lags = r.lag_definition(data=city_data, city=city_name)

    # Setting a pandas.DataFrame to begin modelling
    data_model = r.dataframe_model(data=city_data,
                                   city=city_name,
                                   final_lags=lags,
                                   sectors_df=city_sectors)

    data_predict = r.predict_data(data=city_data,
                                  city=city_name,
                                  final_lags=lags,
                                  sectors_df=city_sectors)
    # Having data to predict

    return data_model, data_predict


def cv(model, config_name, verbose, yearly, city_name):
    with open(config_name, 'r') as f:
        config = yaml.load(f)
        locked_cut = config['LOCKED_BOX_CUT']

    data_model, data_predict = data_preparation(config_name=config_name,
                                                yearly=yearly,
                                                city_name=city_name)
    predictors = [variable for variable in data_model.columns if variable != 'response']

    int_locked_cut = int(np.round(len(data_model) * locked_cut))

    locked_box_data = deepcopy(data_model.iloc[- int_locked_cut:])

    first_lock_index = deepcopy(data_model.index[data_model.index == locked_box_data.index[0]])
    test_index_limit = np.where(data_model.index == first_lock_index[0])[0][0]

    test_set = data_model.iloc[np.arange(test_index_limit-len(locked_box_data), test_index_limit)]
    train_param = np.arange(0, np.where(data_model.index == test_set.index[0])[0][0])

    train_set = data_model.iloc[train_param]
    if model == 'all' and not yearly:
        svm, rf, xgb, lasso, ridge = im.model_party(train_set,
                                                    test_set,
                                                    predictors,
                                                    locked_box_data,
                                                    verbose,
                                                    config_name,
                                                    yearly)
        return svm, rf, xgb, lasso, ridge, data_model, data_predict

    elif model == "all" and yearly:
        if verbose:
            lasso, ridge = im.model_party(train_set,
                                          test_set,
                                          predictors,
                                          locked_box_data,
                                          verbose,
                                          config_name,
                                          yearly)
        return lasso, ridge, data_model, data_predict

    elif model == 'SVM':
        svm = im.svm(train_set, test_set, predictors, locked_box_data, verbose, config_name)
        return svm, data_model, data_predict
    elif model == 'randomForest':
        rf = im.rf(train_set, test_set, predictors, locked_box_data, verbose, config_name)
        return rf, data_model, data_predict
    elif model == 'xgb':
        xgb = im.xgb(train_set, test_set, predictors, locked_box_data, verbose, config_name)
        return xgb, data_model, data_predict
    elif model == 'lasso':
        lasso = im.lasso(train_set, test_set, predictors, locked_box_data, verbose, config_name, yearly)
        return lasso, data_model, data_predict
    elif model == 'ridge':
        ridge = im.ridge(train_set, test_set, predictors, locked_box_data, verbose, config_name, yearly)
        return ridge, data_model, data_predict


def bby_model(model, config_name, verbose, yearly, city):
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge

    if verbose:
        print('Working on model')

    if model == 'all' and not yearly:
        svm, rf, xgb, lasso, ridge, data_model, data_predict = cv(model=model,
                                                                  config_name=config_name,
                                                                  verbose=verbose,
                                                                  yearly=yearly,
                                                                  city_name=city)
    elif model == 'all' and yearly:
        lasso, ridge, data_model, data_predict = cv(model=model,
                                                    config_name=config_name,
                                                    verbose=verbose,
                                                    yearly=yearly,
                                                    city_name=city)
    else:
        model, data_model, data_predict = cv(model, config_name, verbose=verbose, yearly=yearly)

    predictors = [variable for variable in data_model.columns if variable != 'response']

    with open(config_name, 'r') as f:
        config = yaml.load(f)

    dates = data_model.index
    if not yearly:
        list_dates = [dates[i] if i % 6 == 0 else '' for i in range(len(dates))]
        marker = None
        month_to_predict = data_predict.tail(1).month.values[0]
        last_real_data = data_model.tail(1).index.values[0]
    else:
        last_real_data = data_model.tail(1).index[0]
        month_to_predict = str(int(last_real_data.split('_')[0]) + 1) + '_diciembre'
        marker = 'o'


    values = dict()

    mean = lasso[['TEST_MeanAE', 'TEST_MedianAE', 'TEST_RMSE']].mean(axis=1)
    values['lasso'] = mean.loc[mean.values == min(mean.values)].values[0]

    mean = ridge[['TEST_MeanAE', 'TEST_MedianAE', 'TEST_RMSE']].mean(axis=1)
    values['ridge'] = mean.loc[mean.values == min(mean.values)].values[0]

    if model == 'all' and not yearly:
        mean = svm[['TEST_MeanAE', 'TEST_MedianAE', 'TEST_RMSE']].mean(axis=1)
        values['svm'] = mean.loc[mean.values == min(mean.values)].values[0]

        mean = rf[['TEST_MeanAE', 'TEST_MedianAE', 'TEST_RMSE']].mean(axis=1)
        values['rf'] = mean.loc[mean.values == min(mean.values)].values[0]

        mean = xgb[['TEST_MeanAE', 'TEST_MedianAE', 'TEST_RMSE']].mean(axis=1)
        values['xgb'] = mean.loc[mean.values == min(mean.values)].values[0]

    best_model = min(values, key=values.get)
    df_model = locals().get(best_model)
    params_model = df_model.loc[df_model[['TEST_MeanAE', 'TEST_MedianAE', 'TEST_RMSE']].mean(axis=1) ==
                                values.get(best_model)]

    params_model = params_model.index[0].split('-')
    params = {}
    if best_model == 'rf':
        params = {'n_estimators': None,
                  'max_depth': None}
    elif best_model == 'svm':
        params = {'C': None,
                  'degree': None,
                  'kernel': 'linear'}
    elif best_model == 'xgb':
        params = {'learning_rate': None,
                  'max_depth': None,
                  'subsample': None,
                  'n_estimators': 1000}
    elif best_model == 'lasso' or best_model == 'ridge':
        params = {'alpha': None,
                  'fit_intercept': True,
                  'normalize': True,
                  'degree': None}

    keys = params.keys()

    params_floats = {}
    for param in params_model:
        # print param
        try:
            float(param)
            params_floats.update({prev: float(param)})
        except:
            params_floats.update({param: None})
            prev = param
            # print 'except'
            # print prev

    params_floats = {k: v for k, v in params_floats.items() if v}
    if len(params_floats) != keys:
        for key in params.keys():
            if params.get(key) is not None:
                params_floats.update({key: params.get(key)})

    response_variable = deepcopy(data_model['response'])
    predictors = deepcopy(data_model[predictors])

    if best_model == 'svm':
        if params_floats.get('degree') != 1:
            params_floats['kernel'] = 'poly'
        model = SVR(**params_floats)
    elif best_model == 'rf':
        params_floats['n_estimators'] = int(params_floats['n_estimators'])
        model = RandomForestRegressor(**params_floats)
    elif best_model == 'xgb':
        params_floats['max_depth'] = int(params_floats['max_depth'])
        model = XGBRegressor(**params_floats)
    elif best_model == 'lasso' or best_model == 'ridge':
        poly = PolynomialFeatures(degree=int(params_floats.get('degree')), include_bias=False)
        predictors = poly.fit_transform(predictors)
        params_floats.pop('degree', None)
        data_predict = poly.fit_transform(data_predict)
        if best_model == 'lasso':
            model = Lasso(**params_floats)
        else:
            model = Ridge(**params_floats)

    model.fit(predictors, response_variable)
    # if not yearly:
    prediction = model.predict(data_predict)[0]
    mae = values.get(best_model)
    chosen_model = best_model

    if best_model not in ['lasso', 'ridge']:
        previous = data_predict.iloc[-1, 0]
        print('Employment of %s, previous of %s, mean absolute error expected %s. CHhosen model %s bby' % (
           prediction, previous, mae, chosen_model))
    else:
        previous =  data_predict[0][0]
        print('Employment of %s, previous of %s, mean absolute error expected %s. CHhosen model %s bby' % (
            prediction, previous, mae, chosen_model))

    predictors = [variable for variable in data_model.columns if variable != 'response']

    configuration = {'generation_time': datetime.datetime.today().strftime("%Y-%m-%d"),
                     'city': str(city),
                     'prediction_next_month': float(prediction),
                     'previous_employees': float(previous),
                     'expected_mae': float(mae),
                     'chosen_model': str(best_model),
                     'month_to_predict': str(month_to_predict),
                     'last_available_data': str(last_real_data)}

    with open(city + '_config.json', 'wb') as f:
        json.dump(configuration, f)

    return prediction, model, best_model, data_model, data_predict, predictors, configuration


def parse_arguments():
    parser = argparse.ArgumentParser(description='City monthly modelling')
    parser.add_argument("model", type=str,
                        help='Model to choose. Choose "all" to enter model party mode')
    parser.add_argument("config_name", type=str,
                        help='Name of the config file')
    parser.add_argument("-mcv", dest='make_cv',
                        help="Require code to CV through hyper parameters (default : False)",
                        action="store_true")
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-f", "--forecast",
                        help="make 12-month forecast",
                        action="store_true")
    parser.add_argument("-y", "--yearly",
                        help="Require yearly procedure",
                        action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    make_cv = args.make_cv
    verbose = args.verbose
    config_name = args.config_name
    forecast = args.forecast
    yearly = args.yearly

    if yearly:
        import utils_yearly as r
    else:
        import utils as r

    with open(config_name, 'r') as f:
        config = yaml.load(f)
        cities = config['CITY']

    json_list = []
    for city in cities:
        print("On city %s" % city)
        if make_cv:
            if args.verbose:
                print('Entering CV modelling on %s ' % args.model)

            cv(model=args.model,
               config_name=config_name,
               verbose=verbose,
               yearly=yearly,
               city_name=city)

        else:
            if args.model != 'all' and not make_cv:
                print('Sorry, I will check all models because ')
            if not verbose:
                print('Patience will be rewarded')
            prediction, model, best_model, \
            data_model, data_predict, predictors, \
            configuration = bby_model(model='all',
                                      config_name=config_name,
                                      verbose=verbose,
                                      yearly=yearly,
                                      city=city)

            json_list.append(configuration)

    if not yearly:
        name = 'configuration'
    else:
        name = 'configuration_yearly'
    with open(name + '.json', 'wb') as f:
        json.dump(json_list, f)

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('ggplot')


def errors_(test, lock, y_predict, y_predict_l):
    meanae = mean_absolute_error(test, y_predict)
    medianae = median_absolute_error(test, y_predict)
    mse = mean_squared_error(test, y_predict) ** 0.5

    meanae_l = mean_absolute_error(lock, y_predict_l)
    medianae_l = median_absolute_error(lock, y_predict_l)
    mse_l = mean_squared_error(lock, y_predict_l) ** 0.5

    pablos_test = abs(sum(test) - sum(y_predict)) / sum(test)
    pablos_lock = abs(sum(lock) - sum(y_predict_l)) / sum(lock)
    return [meanae, medianae, mse, pablos_test, meanae_l, medianae_l, mse_l,  pablos_lock]


def plot_model(model, param_results):
    best_model = param_results.loc[param_results.LOCK_MedianAE == param_results.LOCK_MedianAE.min()].index

    pdf = PdfPages(model + 'general_metrics.pdf')
    plt.figure()
    param_results.plot(marker='o')
    plt.xticks(range(len(param_results)), param_results.index, rotation=90)
    plt.title('Metrics of error with ' + model)
    plt.xlabel('Params')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='best')
    pdf.savefig()
    pdf.close()

    pdf = PdfPages(model + 'best_model.pdf')
    plt.figure()
    param_results.loc[best_model].plot(marker='o')
    plt.xticks(range(len(best_model)), best_model, rotation=90)
    plt.title('Metrics of error with ' + model)
    plt.xlabel('Params')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='best')
    pdf.savefig()
    pdf.close()


# CV models
def svm(train_set, test_set, predictors, locked_box_data, verbose, config_name, to_plot=True):
    with open(config_name, "r") as f:
        config = yaml.load(f)
        c = config['C_PARAMETER']
        degree = config['DEGREE']

    train_x = train_set[predictors]
    train_y = train_set['response']
    test_x = test_set[predictors]
    kernel = ['linear', 'rbf']
    param_results = pd.DataFrame(data=None,
                                 index=['C-' + str(the_c) +
                                        '-degree-' + str(deg) + '-kernel-' +
                                        str(ker)
                                        for the_c in c
                                        for ker in kernel
                                        for deg in degree
                                        ],
                                 columns=[the_type + '_' + error for the_type in ['TEST', 'LOCK']
                                          for error in ['MeanAE', 'MedianAE', 'RMSE', 'Rel_diff']])

    # Dropping degrees on linear kernel
    for i in param_results.index:
        if 'linear' in i and i.split('-')[3] != '1':
            param_results.drop(i, inplace=True)

    for the_c in c:
        for ker in kernel:
            for deg in degree:
                if verbose:
                    print('Currently on c %s, kernel %s and degree %s' % (the_c, ker, deg))
                svm = SVR(C=the_c, kernel=ker, degree=deg, verbose=False, max_iter=500)
                svm.fit(train_x, train_y)
                y_predict = svm.predict(test_x)
                y_predict_l = svm.predict(locked_box_data[predictors])
                param_results.loc['C-' + str(the_c) + '-degree-' + str(deg) + '-kernel-' + str(ker)] = \
                    errors_(test_set['response'], locked_box_data['response'], y_predict, y_predict_l)

    best_model = param_results.loc[param_results.LOCK_MedianAE == param_results.LOCK_MedianAE.min()].index

    if to_plot:
        plot_model('SVM', param_results)

    if verbose:
        if len(best_model) == 1:
            print('Best model on %s' % best_model[0])
        else:
            print('Best model on %s' % best_model[0])
    return param_results, best_model


def rf(train_set, test_set, predictors, locked_box_data, verbose,  config_name, to_plot=True):
    with open(config_name, "r") as f:
        config = yaml.load(f)
    n_trees = config['N_TREES']
    n_vars = config['N_VARS']

    train_x = train_set[predictors]
    train_y = train_set['response']
    test_x = test_set[predictors]
    while max(n_vars) > train_x.shape[1]:
        if verbose:
            print('WARNING: Bad hombre on N_VARS')
        n_vars.remove(max(n_vars))

    param_results = pd.DataFrame(data=None,
                                 index=['n_estimators-' + str(tree) +
                                        '-max_depth-' + str(the_var)
                                        for tree in n_trees
                                        for the_var in n_vars
                                        ],
                                 columns=[the_type + '_' + error for the_type in ['TEST', 'LOCK']
                                          for error in ['MeanAE', 'MedianAE', 'RMSE', 'Rel_diff']])

    for tree in n_trees:
        for vars in n_vars:
            if verbose:
                print('Currently on trees %s and vars %s' % (tree, vars))
                rf = RandomForestRegressor(n_estimators=tree, max_depth=vars, n_jobs=-1)
                rf.fit(train_x, train_y)

                y_predict = rf.predict(test_x)
                y_predict_l = rf.predict(locked_box_data[predictors])

                param_results.loc['n_estimators-' + str(tree) + '-max_depth-' + str(vars)] = \
                    errors_(test_set['response'],
                            locked_box_data['response'],
                            y_predict,
                            y_predict_l)

    best_model = param_results.loc[param_results.LOCK_MedianAE == param_results.LOCK_MedianAE.min()].index
    if to_plot:
        plot_model('randomForest', param_results)

    if verbose:
        if len(best_model) == 1:
            print('Best model on %s' % best_model[0])
        else:
            print('Best model on %s' % best_model[0])

    return param_results, best_model


def xgb(train_set, test_set, predictors, locked_box_data, verbose,  config_name, to_plot=True):
    with open(config_name, "r") as f:
        config = yaml.load(f)
        rates = config['LEARNING_RATES']
        m_depths = config['MAX_DEPTH']
        subsams = config['SUBSAMPLE']

    train_x = train_set[predictors]
    train_y = train_set['response']
    test_x = test_set[predictors]
    param_results = pd.DataFrame(data=0,
                                 index=['learning_rate-' + str(lr) +
                                        '-max_depth-' + str(md) +
                                        '-subsample-' + str(ss)
                                        for lr in rates
                                        for md in m_depths
                                        for ss in subsams
                                        ],
                                 columns=[the_type + '_' + error for the_type in ['TEST', 'LOCK']
                                          for error in ['MeanAE', 'MedianAE', 'RMSE', 'Rel_diff']])
    param_results = param_results[~param_results.index.duplicated(keep='first')]

    for lr in rates:
        for md in m_depths:
            for ss in subsams:
                if verbose:
                    print('Currenty at learning rate %s, max depth of %s and subsample of %s' % (lr, md, ss))
                xgb = XGBRegressor(nthread=-1, learning_rate=lr, max_depth=md, subsample=ss,
                                   n_estimators=1000)
                xgb.fit(train_x, train_y)

                y_predict = xgb.predict(test_x)
                y_predict_l = xgb.predict(locked_box_data[predictors])

                c_index = 'learning_rate-' + str(lr) + '-max_depth-' + str(md) + '-subsample-' + str(ss)
                param_results.loc[c_index] = errors_(test_set['response'],
                                                     locked_box_data['response'],
                                                     y_predict,
                                                     y_predict_l)

    best_model = param_results.loc[param_results.LOCK_MedianAE == param_results.LOCK_MedianAE.min()].index
    if to_plot:
        plot_model('xgb', param_results)
    if verbose:
        if len(best_model) == 1:
            print('Best model on %s' % best_model[0])
        else:
            print('Best model on %s' % best_model[0])
    return param_results, best_model


def lasso(train_set, test_set, predictors, locked_box_data, verbose,  config_name, yearly, to_plot=True):
    with open(config_name, "r") as f:
        config = yaml.load(f)
        alpha_term = config['ALPHA_REG_TERM']
        degrees = config['DEGREE_REGS']

    if not yearly:
        degrees = [1]

    train_x = train_set[predictors]
    train_y = train_set['response']
    test_x = test_set[predictors]
    locked_x = locked_box_data[predictors]
    param_results = pd.DataFrame(data=0,
                                 index=['degree-' + str(degree) +
                                        '-alpha-' + str(the_alpha)
                                        for degree in degrees for the_alpha in alpha_term],
                                 columns=[the_type + '_' + error for the_type in ['TEST', 'LOCK']
                                          for error in ['MeanAE', 'MedianAE', 'RMSE', 'Rel_diff']])
    param_results = param_results[~param_results.index.duplicated(keep='first')]

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        train_x_poly = poly.fit_transform(train_x)
        test_x_poly = poly.fit_transform(test_x)
        locked_box_data_poly = poly.fit_transform(locked_x)
        for the_alpha in alpha_term:
            if verbose:
                print("Currently on degree %s and L1 norm penalization term %s" % (degree, the_alpha))
            model = Lasso(alpha=the_alpha, normalize=True, fit_intercept=True)
            model.fit(train_x_poly, train_y)
            y_predict = model.predict(test_x_poly)
            y_predict_l = model.predict(locked_box_data_poly)

            c_index = 'degree-' + str(degree) + '-alpha-' + str(the_alpha)
            param_results.loc[c_index] = errors_(test_set['response'],
                                                 locked_box_data['response'],
                                                 y_predict,
                                                 y_predict_l)

    best_model = param_results.loc[param_results.LOCK_MedianAE == param_results.LOCK_MedianAE.min()].index
    if to_plot:
        plot_model('lasso', param_results)
    if verbose:
        if len(best_model) == 1:
            print('Best lasso model on %s' % best_model[0])
        else:
            print('Best lasso model on %s' % best_model[0])
    return param_results, best_model


def ridge(train_set, test_set, predictors, locked_box_data, verbose,  config_name, yearly, to_plot=True):
    with open(config_name, "r") as f:
        config = yaml.load(f)
        alpha_term = config['ALPHA_REG_TERM']
        degrees = config['DEGREE_REGS']

    if not yearly:
        degrees = [1]

    train_x = train_set[predictors]
    train_y = train_set['response']
    test_x = test_set[predictors]
    locked_x = locked_box_data[predictors]
    param_results = pd.DataFrame(data=0,
                                 index=['degree-' + str(degree) +
                                        '-alpha-' + str(the_alpha)
                                        for degree in degrees for the_alpha in alpha_term],
                                 columns=[the_type + '_' + error for the_type in ['TEST', 'LOCK']
                                          for error in ['MeanAE', 'MedianAE', 'RMSE', 'Rel_diff']])
    param_results = param_results[~param_results.index.duplicated(keep='first')]

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        train_x_poly = poly.fit_transform(train_x)
        test_x_poly = poly.fit_transform(test_x)
        locked_box_data_poly = poly.fit_transform(locked_x)
        for the_alpha in alpha_term:
            if verbose:
                print("Currently on degree %s and L2 norm penalization term %s" %(degree, the_alpha))
            model = Ridge(alpha=the_alpha, normalize=True, fit_intercept=True)
            model.fit(train_x_poly, train_y)
            y_predict = model.predict(test_x_poly)
            y_predict_l = model.predict(locked_box_data_poly)

            c_index = 'degree-' + str(degree) + '-alpha-' + str(the_alpha)
            param_results.loc[c_index] = errors_(test_set['response'],
                                                 locked_box_data['response'],
                                                 y_predict,
                                                 y_predict_l)

    best_model = param_results.loc[param_results.LOCK_MedianAE == param_results.LOCK_MedianAE.min()].index
    if to_plot:
        plot_model('ridge', param_results)
    if verbose:
        if len(best_model) == 1:
            print('Best ridge model on %s' % best_model[0])
        else:
            print('Best ridge model on %s' % best_model[0])
    return param_results, best_model


def model_party(train_set, test_set, predictors, locked_box_data, verbose, config_name, yearly):
    pdf = PdfPages('model_party_general_metrics.pdf')
    plt.figure()

    _4, best_lasso = lasso(train_set, test_set, predictors, locked_box_data, verbose, config_name, yearly)
    _5, best_ridge = ridge(train_set, test_set, predictors, locked_box_data, verbose, config_name, yearly)

    _4.plot(marker='o')
    plt.xticks(range(len(_4)), _4.index, rotation=90)
    plt.title('Metrics of error with lasso')
    plt.xlabel('Params')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='best')
    pdf.savefig()

    _5.plot(marker='o')
    plt.xticks(range(len(_5)), _5.index, rotation=90)
    plt.title('Metrics of error with ridge')
    plt.xlabel('Params')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='best')
    pdf.savefig()

    if not yearly:

        _1, best_svm = svm(train_set, test_set, predictors, locked_box_data, verbose, config_name, yearly)
        _2, best_rf = rf(train_set, test_set, predictors, locked_box_data, verbose, config_name, yearly)
        _3, best_xgb = xgb(train_set, test_set, predictors, locked_box_data, verbose,  config_name, yearly)

        _1.plot(marker='o')
        plt.xticks(range(len(_1)), _1.index, rotation=90)
        plt.title('Metrics of error with SVM')
        plt.xlabel('Params')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend(loc='best')
        pdf.savefig()

        _2.plot(marker='o')
        plt.xticks(range(len(_2)), _2.index, rotation=90)
        plt.title('Metrics of error with randomForest')
        plt.xlabel('Params')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend(loc='best')
        pdf.savefig()

        _3.plot(marker='o')
        plt.xticks(range(len(_3)), _3.index, rotation=90)
        plt.title('Metrics of error with xgb')
        plt.xlabel('Params')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend(loc='best')
        pdf.savefig()

    # else:
    #     _4, best_lasso = lasso(train_set, test_set, predictors, locked_box_data, verbose, config_name, False)
    #     _5, best_ridge = ridge(train_set, test_set, predictors, locked_box_data, verbose, config_name, False)
    #
    #     _4.plot(marker='o')
    #     plt.xticks(range(len(_4)), _4.index, rotation=90)
    #     plt.title('Metrics of error with lasso')
    #     plt.xlabel('Params')
    #     plt.ylabel('Error')
    #     plt.grid(True)
    #     plt.legend(loc='best')
    #     pdf.savefig()
    #
    #     _5.plot(marker='o')
    #     plt.xticks(range(len(_5)), _5.index, rotation=90)
    #     plt.title('Metrics of error with ridge')
    #     plt.xlabel('Params')
    #     plt.ylabel('Error')
    #     plt.grid(True)
    #     plt.legend(loc='best')
    #     pdf.savefig()

    pdf.close()

    pdf = PdfPages('model_party_best_model.pdf')
    plt.figure()

    _4.loc[best_lasso].plot(marker='o')
    plt.xticks(range(len(best_lasso)), best_lasso, rotation=90)
    plt.title('Metrics of error with lasso')
    plt.xlabel('Params')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='best')
    pdf.savefig()

    _5.loc[best_ridge].plot(marker='o')
    plt.xticks(range(len(best_ridge)), best_ridge, rotation=90)
    plt.title('Metrics of error with ridge')
    plt.xlabel('Params')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='best')
    pdf.savefig()

    if not yearly:

        _1.loc[best_svm].plot(marker='o')
        plt.xticks(range(len(best_svm)), best_svm, rotation=90)
        plt.title('Metrics of error with SVM')
        plt.xlabel('Params')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend(loc='best')
        pdf.savefig()

        _2.loc[best_rf].plot(marker='o')
        plt.xticks(range(len(best_rf)), best_rf, rotation=90)
        plt.title('Metrics of error with randomForest')
        plt.xlabel('Params')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend(loc='best')
        pdf.savefig()

        _3.loc[best_xgb].plot(marker='o')
        plt.xticks(range(len(best_xgb)), best_xgb, rotation=90)
        plt.title('Metrics of error with xgb')
        plt.xlabel('Params')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend(loc='best')
        pdf.savefig()

    # else:
    #     _4.loc[best_lasso].plot(marker='o')
    #     plt.xticks(range(len(best_lasso)), best_lasso, rotation=90)
    #     plt.title('Metrics of error with lasso')
    #     plt.xlabel('Params')
    #     plt.ylabel('Error')
    #     plt.grid(True)
    #     plt.legend(loc='best')
    #     pdf.savefig()
    #
    #     _5.loc[best_ridge].plot(marker='o')
    #     plt.xticks(range(len(best_ridge)), best_ridge, rotation=90)
    #     plt.title('Metrics of error with ridge')
    #     plt.xlabel('Params')
    #     plt.ylabel('Error')
    #     plt.grid(True)
    #     plt.legend(loc='best')
    #     pdf.savefig()

    pdf.close()

    if not yearly:
        return _1, _2, _3, _4, _5
    return _4, _5


# Production models


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
import base64
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import warnings
import seaborn as sns
import networkx as nx
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering



def predicciones_generales(prediccion_escoger):
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    from datetime import datetime, timedelta
    d = datetime.today() - timedelta(days=1)
    fecha_actual = d.strftime('%m-%d-%Y')
    print(fecha_actual)
    fecha_actual2 = d.strftime('%Y-%m-%d')
    print(fecha_actual2)
    latest_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/" + fecha_actual + ".csv")
    us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+ fecha_actual + ".csv")   
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []
    total_deaths = [] 
    mortality_rate = []
    recovery_rate = [] 
    total_recovered = [] 
    total_active = [] 

    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()
        
        # confirmed, deaths, recovered, and active
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        total_recovered.append(recovered_sum)
        total_active.append(confirmed_sum-death_sum-recovered_sum)
        
        # calculate rates
        mortality_rate.append(death_sum/confirmed_sum)
        recovery_rate.append(recovered_sum/confirmed_sum)
    def daily_increase(data):
        d = [] 
        for i in range(len(data)):
            if i == 0:
                d.append(data[0])
            else:
                d.append(data[i]-data[i-1])
        return d 

    def moving_average(data, window_size):
        moving_average = []
        for i in range(len(data)):
            if i + window_size < len(data):
                moving_average.append(np.mean(data[i:i+window_size]))
            else:
                moving_average.append(np.mean(data[i:len(data)]))
        return moving_average

    window = 7

    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg= moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)

    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)

    world_daily_recovery = daily_increase(total_recovered)
    world_recovery_avg = moving_average(total_recovered, window)
    world_daily_recovery_avg = moving_average(world_daily_recovery, window)

    world_active_avg = moving_average(total_active, window)
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    total_recovered = np.array(total_recovered).reshape(-1, 1)
    days_in_future = 50
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-50]

    import datetime
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.05, shuffle=False)

    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    #plt.plot(y_test_confirmed)
    #plt.plot(svm_test_pred)
    #plt.legend(['Test Data', 'SVM Predictions'])

    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
    bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)

    #plt.plot(y_test_confirmed)
    #plt.plot(test_linear_pred)
    #plt.legend(['Test Data', 'Polynomial Regression Predictions'])

    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                    'normalize' : normalize}

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
    bayesian_search.best_params_
    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    #plt.plot(y_test_confirmed)
    #plt.plot(test_bayesian_pred)
    #plt.legend(['Test Data', 'Predicciones de Cresta Bayesianas'])

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(bayesian_search, y_test_confirmed, test_bayesian_pred, cv=4)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*-1, scores.std() * 2))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train_confirmed, y_train_confirmed)
    print(clf.coef_)
    print(clf.intercept_)
    testlassopredict = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])


    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, y_train_confirmed, testlassopredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    lasso_pred = clf.predict(future_forcast)

    from sklearn.linear_model import LassoCV
    from sklearn.datasets import make_regression
    reg = LassoCV(cv=5, random_state=0).fit(X_train_confirmed, y_train_confirmed)
    #reg.score(X, y)
    testlassoCVpredict = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])

    scores = cross_val_score(reg, X_train_confirmed, testlassoCVpredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.linear_model import ElasticNetCV
    from sklearn.datasets import make_regression
    regr = ElasticNetCV(cv=50, random_state=0)
    regr.fit(X_train_confirmed, y_train_confirmed)
    print(regr.alpha_)
    print(regr.intercept_)
    test_elasticnet = regr.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(test_elasticnet)
    #plt.legend(['Original Data', 'ElasticNetCV Predictions'])

    from sklearn import linear_model
    reg = linear_model.BayesianRidge()
    reg.fit(X_train_confirmed, y_train_confirmed)
    pred_bayesian_ridge = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Bayesian Ridge Predictions'])


    scores = cross_val_score(regr, X_train_confirmed, test_elasticnet, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000000000000)
    clf.fit(X_train_confirmed, y_train_confirmed)
    predictneural = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(predictneural)
    #plt.legend(['Original Data', 'Neural Network Predictions'])

    neuronal_pred = clf.predict(future_forcast)


    scores = cross_val_score(clf, X_train_confirmed, predictneural, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    def country_plot(x, y1, y2, y3, y4, country):
        # window is set as 14 in in the beginning of the notebook 
        confirmed_avg = moving_average(y1, window)
        confirmed_increase_avg = moving_average(y2, window)
        death_increase_avg = moving_average(y3, window)
        recovery_increase_avg = moving_average(y4, window)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure(figsize=(16, 10))
        plt.plot(x, y1)
        plt.plot(x, confirmed_avg, color='red', linestyle='dashed')
        plt.legend(['{} Confirmed Cases'.format(country), 'Moving Average {} Days'.format(window)], prop={'size': 20})
        plt.title('{} Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y2)
        plt.plot(x, confirmed_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Cases'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y3)
        plt.plot(x, death_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Deaths'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Deaths'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y4)
        plt.plot(x, recovery_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Recoveries'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        
    # helper function for getting country's cases, deaths, and recoveries        
    def get_country_info(country_name):
        country_cases = []
        country_deaths = []
        country_recoveries = []  
        
        for i in dates:
            country_cases.append(confirmed_df[confirmed_df['Country/Region']==country_name][i].sum())
            country_deaths.append(deaths_df[deaths_df['Country/Region']==country_name][i].sum())
            country_recoveries.append(recoveries_df[recoveries_df['Country/Region']==country_name][i].sum())
        return (country_cases, country_deaths, country_recoveries)
        
        
    def country_visualizations(country_name):
        country_info = get_country_info(country_name)
        country_cases = country_info[0]
        country_deaths = country_info[1]
        country_recoveries = country_info[2]
        
        country_daily_increase = daily_increase(country_cases)
        country_daily_death = daily_increase(country_deaths)
        country_daily_recovery = daily_increase(country_recoveries)
        
        country_plot(adjusted_dates, country_cases, country_daily_increase, country_daily_death, country_daily_recovery, country_name)

    countries = ['Albania',
    'Algeria',
    'Andorra',
    'Angola',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bhutan',
    'Bolivia',
    'Bosnia and Herzegovina',
    'Botswana',
    'Brazil',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burma',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Central African Republic',
    'Chad',
    'Chile',
    'Colombia',
    'Comoros',
    'Congo (Brazzaville)',
    'Congo (Kinshasa)',
    'Costa Rica',
    'Croatia',
    'Cuba',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Diamond Princess',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea',
    'Estonia',
    'Eswatini',
    'Ethiopia',
    'Fiji',
    'Finland',
    'France',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Greece',
    'Grenada',
    'Guatemala',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Holy See',
    'Honduras',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Ireland',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Korea, South',
    'Kosovo',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'MS Zaandam',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Mauritania',
    'Mauritius',
    'Mexico',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Mozambique',
    'Namibia',
    'Nepal',
    'Netherlands',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'North Macedonia',
    'Norway',
    'Oman',
    'Pakistan',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Qatar',
    'Romania',
    'Russia',
    'Rwanda',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'San Marino',
    'Sao Tome and Principe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Slovakia',
    'Slovenia',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'Sudan',
    'Suriname',
    'Sweden',
    'Switzerland',
    'Syria',
    'Taiwan*',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Trinidad and Tobago',
    'Tunisia',
    'Turkey',
    'US',
    'Uganda',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'Uruguay',
    'Uzbekistan',
    'Venezuela',
    'Vietnam',
    'West Bank and Gaza',
    'Western Sahara',
    'Yemen',
    'Zambia',
    'Zimbabwe',
    ]

    def plot_predictions(x, y, pred, algo_name, color):
        plt.style.use(['dark_background'])
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.title('COVID-19', size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('Cantidad de Casos', size=30)
        plt.legend(['Casos Confirmados', algo_name], prop={'size': 20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        nombre_archivo = " predicciones_covid_" + algo_name  + '.png'
        plt.savefig(nombre_archivo)
        #plt.show()

    if (prediccion_escoger == "RedesNeuronales"):
        plot_predictions(adjusted_dates, world_cases, neuronal_pred , 'Predicciones de Redes Neuronales a nivel general', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Redes Neuronales a nivel general' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "Lasso"):
        plot_predictions(adjusted_dates, world_cases, lasso_pred, 'Predicciones de regresión de Lasso a nivel general', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de Lasso a nivel general' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "RegresionCrestaBayesiana"):
        plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7

def predicciones_mortalidad_generales(prediccion_escoger):
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    from datetime import datetime, timedelta
    d = datetime.today() - timedelta(days=1)
    fecha_actual = d.strftime('%m-%d-%Y')
    print(fecha_actual)
    fecha_actual2 = d.strftime('%Y-%m-%d')
    print(fecha_actual2)
    latest_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/" + fecha_actual + ".csv")
    us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+ fecha_actual + ".csv")   
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []
    total_deaths = [] 
    mortality_rate = []
    recovery_rate = [] 
    total_recovered = [] 
    total_active = [] 

    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()
        
        # confirmed, deaths, recovered, and active
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        total_recovered.append(recovered_sum)
        total_active.append(confirmed_sum-death_sum-recovered_sum)
        
        # calculate rates
        mortality_rate.append(death_sum/confirmed_sum)
        recovery_rate.append(recovered_sum/confirmed_sum)
    def daily_increase(data):
        d = [] 
        for i in range(len(data)):
            if i == 0:
                d.append(data[0])
            else:
                d.append(data[i]-data[i-1])
        return d 

    def moving_average(data, window_size):
        moving_average = []
        for i in range(len(data)):
            if i + window_size < len(data):
                moving_average.append(np.mean(data[i:i+window_size]))
            else:
                moving_average.append(np.mean(data[i:len(data)]))
        return moving_average

    window = 7

    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg= moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)

    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)

    world_daily_recovery = daily_increase(total_recovered)
    world_recovery_avg = moving_average(total_recovered, window)
    world_daily_recovery_avg = moving_average(world_daily_recovery, window)

    world_active_avg = moving_average(total_active, window)
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    total_recovered = np.array(total_recovered).reshape(-1, 1)
    days_in_future = 50
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-50]

    import datetime
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.05, shuffle=False)

    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    #plt.plot(y_test_confirmed)
    #plt.plot(svm_test_pred)
    #plt.legend(['Test Data', 'SVM Predictions'])

    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
    bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)

    #plt.plot(y_test_confirmed)
    #plt.plot(test_linear_pred)
    #plt.legend(['Test Data', 'Polynomial Regression Predictions'])

    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                    'normalize' : normalize}

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
    bayesian_search.best_params_
    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    #plt.plot(y_test_confirmed)
    #plt.plot(test_bayesian_pred)
    #plt.legend(['Test Data', 'Predicciones de Cresta Bayesianas'])

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(bayesian_search, y_test_confirmed, test_bayesian_pred, cv=4)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*-1, scores.std() * 2))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train_confirmed, y_train_confirmed)
    print(clf.coef_)
    print(clf.intercept_)
    testlassopredict = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])


    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, y_train_confirmed, testlassopredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    lasso_pred = clf.predict(future_forcast)

    from sklearn.linear_model import LassoCV
    from sklearn.datasets import make_regression
    reg = LassoCV(cv=5, random_state=0).fit(X_train_confirmed, y_train_confirmed)
    #reg.score(X, y)
    testlassoCVpredict = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])

    scores = cross_val_score(reg, X_train_confirmed, testlassoCVpredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.linear_model import ElasticNetCV
    from sklearn.datasets import make_regression
    regr = ElasticNetCV(cv=50, random_state=0)
    regr.fit(X_train_confirmed, y_train_confirmed)
    print(regr.alpha_)
    print(regr.intercept_)
    test_elasticnet = regr.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(test_elasticnet)
    #plt.legend(['Original Data', 'ElasticNetCV Predictions'])

    from sklearn import linear_model
    reg = linear_model.BayesianRidge()
    reg.fit(X_train_confirmed, y_train_confirmed)
    pred_bayesian_ridge = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Bayesian Ridge Predictions'])


    scores = cross_val_score(regr, X_train_confirmed, test_elasticnet, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000000000000)
    clf.fit(X_train_confirmed, y_train_confirmed)
    predictneural = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(predictneural)
    #plt.legend(['Original Data', 'Neural Network Predictions'])

    neuronal_pred = clf.predict(future_forcast)


    scores = cross_val_score(clf, X_train_confirmed, predictneural, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    def country_plot(x, y1, y2, y3, y4, country):
        # window is set as 14 in in the beginning of the notebook 
        confirmed_avg = moving_average(y1, window)
        confirmed_increase_avg = moving_average(y2, window)
        death_increase_avg = moving_average(y3, window)
        recovery_increase_avg = moving_average(y4, window)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure(figsize=(16, 10))
        plt.plot(x, y1)
        plt.plot(x, confirmed_avg, color='red', linestyle='dashed')
        plt.legend(['{} Confirmed Cases'.format(country), 'Moving Average {} Days'.format(window)], prop={'size': 20})
        plt.title('{} Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y2)
        plt.plot(x, confirmed_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Cases'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y3)
        plt.plot(x, death_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Deaths'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Deaths'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y4)
        plt.plot(x, recovery_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Recoveries'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        
    # helper function for getting country's cases, deaths, and recoveries        
    def get_country_info(country_name):
        country_cases = []
        country_deaths = []
        country_recoveries = []  
        
        for i in dates:
            country_cases.append(confirmed_df[confirmed_df['Country/Region']==country_name][i].sum())
            country_deaths.append(deaths_df[deaths_df['Country/Region']==country_name][i].sum())
            country_recoveries.append(recoveries_df[recoveries_df['Country/Region']==country_name][i].sum())
        return (country_cases, country_deaths, country_recoveries)
        
        
    def country_visualizations(country_name):
        country_info = get_country_info(country_name)
        country_cases = country_info[0]
        country_deaths = country_info[1]
        country_recoveries = country_info[2]
        
        country_daily_increase = daily_increase(country_cases)
        country_daily_death = daily_increase(country_deaths)
        country_daily_recovery = daily_increase(country_recoveries)
        
        country_plot(adjusted_dates, country_cases, country_daily_increase, country_daily_death, country_daily_recovery, country_name)

    countries = ['Albania',
    'Algeria',
    'Andorra',
    'Angola',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bhutan',
    'Bolivia',
    'Bosnia and Herzegovina',
    'Botswana',
    'Brazil',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burma',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Central African Republic',
    'Chad',
    'Chile',
    'Colombia',
    'Comoros',
    'Congo (Brazzaville)',
    'Congo (Kinshasa)',
    'Costa Rica',
    'Croatia',
    'Cuba',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Diamond Princess',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea',
    'Estonia',
    'Eswatini',
    'Ethiopia',
    'Fiji',
    'Finland',
    'France',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Greece',
    'Grenada',
    'Guatemala',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Holy See',
    'Honduras',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Ireland',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Korea, South',
    'Kosovo',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'MS Zaandam',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Mauritania',
    'Mauritius',
    'Mexico',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Mozambique',
    'Namibia',
    'Nepal',
    'Netherlands',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'North Macedonia',
    'Norway',
    'Oman',
    'Pakistan',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Qatar',
    'Romania',
    'Russia',
    'Rwanda',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'San Marino',
    'Sao Tome and Principe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Slovakia',
    'Slovenia',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'Sudan',
    'Suriname',
    'Sweden',
    'Switzerland',
    'Syria',
    'Taiwan*',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Trinidad and Tobago',
    'Tunisia',
    'Turkey',
    'US',
    'Uganda',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'Uruguay',
    'Uzbekistan',
    'Venezuela',
    'Vietnam',
    'West Bank and Gaza',
    'Western Sahara',
    'Yemen',
    'Zambia',
    'Zimbabwe',
    ]

    def plot_predictions(x, y, pred, algo_name, color):
        plt.style.use(['dark_background'])
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.title('COVID-19', size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('Cantidad de Casos', size=30)
        plt.legend(['Casos Confirmados', algo_name], prop={'size': 20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        nombre_archivo = " predicciones_covid_" + algo_name  + '.png'
        plt.savefig(nombre_archivo)
        #plt.show()

    if (prediccion_escoger == "RedesNeuronales"):
        plot_predictions(adjusted_dates, world_cases, neuronal_pred , 'Predicciones de Redes Neuronales a nivel general sobre la mortalidad', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Redes Neuronales a nivel general sobre la mortalidad' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "Lasso"):
        plot_predictions(adjusted_dates, world_cases, lasso_pred, 'Predicciones de regresión de Lasso a nivel general sobre la mortalidad', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de Lasso a nivel general sobre la mortalidad' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "RegresionCrestaBayesiana"):
        plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana sobre la mortalidad', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana sobre la mortalidad' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7

def predicciones_recuperacion_generales(prediccion_escoger):
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    from datetime import datetime, timedelta
    d = datetime.today() - timedelta(days=1)
    fecha_actual = d.strftime('%m-%d-%Y')
    print(fecha_actual)
    fecha_actual2 = d.strftime('%Y-%m-%d')
    print(fecha_actual2)
    latest_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/" + fecha_actual + ".csv")
    us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+ fecha_actual + ".csv")   
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []
    total_deaths = [] 
    mortality_rate = []
    recovery_rate = [] 
    total_recovered = [] 
    total_active = [] 

    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()
        
        # confirmed, deaths, recovered, and active
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        total_recovered.append(recovered_sum)
        total_active.append(confirmed_sum-death_sum-recovered_sum)
        
        # calculate rates
        mortality_rate.append(death_sum/confirmed_sum)
        recovery_rate.append(recovered_sum/confirmed_sum)
    def daily_increase(data):
        d = [] 
        for i in range(len(data)):
            if i == 0:
                d.append(data[0])
            else:
                d.append(data[i]-data[i-1])
        return d 

    def moving_average(data, window_size):
        moving_average = []
        for i in range(len(data)):
            if i + window_size < len(data):
                moving_average.append(np.mean(data[i:i+window_size]))
            else:
                moving_average.append(np.mean(data[i:len(data)]))
        return moving_average

    window = 7

    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg= moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)

    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)

    world_daily_recovery = daily_increase(total_recovered)
    world_recovery_avg = moving_average(total_recovered, window)
    world_daily_recovery_avg = moving_average(world_daily_recovery, window)

    world_active_avg = moving_average(total_active, window)
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    total_recovered = np.array(total_recovered).reshape(-1, 1)
    days_in_future = 50
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-50]

    import datetime
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.05, shuffle=False)

    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    #plt.plot(y_test_confirmed)
    #plt.plot(svm_test_pred)
    #plt.legend(['Test Data', 'SVM Predictions'])

    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
    bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)

    #plt.plot(y_test_confirmed)
    #plt.plot(test_linear_pred)
    #plt.legend(['Test Data', 'Polynomial Regression Predictions'])

    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                    'normalize' : normalize}

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
    bayesian_search.best_params_
    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    #plt.plot(y_test_confirmed)
    #plt.plot(test_bayesian_pred)
    #plt.legend(['Test Data', 'Predicciones de Cresta Bayesianas'])

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(bayesian_search, y_test_confirmed, test_bayesian_pred, cv=4)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*-1, scores.std() * 2))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train_confirmed, y_train_confirmed)
    print(clf.coef_)
    print(clf.intercept_)
    testlassopredict = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])


    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, y_train_confirmed, testlassopredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    lasso_pred = clf.predict(future_forcast)

    from sklearn.linear_model import LassoCV
    from sklearn.datasets import make_regression
    reg = LassoCV(cv=5, random_state=0).fit(X_train_confirmed, y_train_confirmed)
    #reg.score(X, y)
    testlassoCVpredict = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])

    scores = cross_val_score(reg, X_train_confirmed, testlassoCVpredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.linear_model import ElasticNetCV
    from sklearn.datasets import make_regression
    regr = ElasticNetCV(cv=50, random_state=0)
    regr.fit(X_train_confirmed, y_train_confirmed)
    print(regr.alpha_)
    print(regr.intercept_)
    test_elasticnet = regr.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(test_elasticnet)
    #plt.legend(['Original Data', 'ElasticNetCV Predictions'])

    from sklearn import linear_model
    reg = linear_model.BayesianRidge()
    reg.fit(X_train_confirmed, y_train_confirmed)
    pred_bayesian_ridge = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Bayesian Ridge Predictions'])


    scores = cross_val_score(regr, X_train_confirmed, test_elasticnet, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=100000000000000)
    clf.fit(X_train_confirmed, y_train_confirmed)
    predictneural = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(predictneural)
    #plt.legend(['Original Data', 'Neural Network Predictions'])

    neuronal_pred = clf.predict(future_forcast)


    scores = cross_val_score(clf, X_train_confirmed, predictneural, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    def country_plot(x, y1, y2, y3, y4, country):
        # window is set as 14 in in the beginning of the notebook 
        confirmed_avg = moving_average(y1, window)
        confirmed_increase_avg = moving_average(y2, window)
        death_increase_avg = moving_average(y3, window)
        recovery_increase_avg = moving_average(y4, window)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure(figsize=(16, 10))
        plt.plot(x, y1)
        plt.plot(x, confirmed_avg, color='red', linestyle='dashed')
        plt.legend(['{} Confirmed Cases'.format(country), 'Moving Average {} Days'.format(window)], prop={'size': 20})
        plt.title('{} Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y2)
        plt.plot(x, confirmed_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Cases'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y3)
        plt.plot(x, death_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Deaths'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Deaths'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y4)
        plt.plot(x, recovery_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Recoveries'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        
    # helper function for getting country's cases, deaths, and recoveries        
    def get_country_info(country_name):
        country_cases = []
        country_deaths = []
        country_recoveries = []  
        
        for i in dates:
            country_cases.append(confirmed_df[confirmed_df['Country/Region']==country_name][i].sum())
            country_deaths.append(deaths_df[deaths_df['Country/Region']==country_name][i].sum())
            country_recoveries.append(recoveries_df[recoveries_df['Country/Region']==country_name][i].sum())
        return (country_cases, country_deaths, country_recoveries)
        
        
    def country_visualizations(country_name):
        country_info = get_country_info(country_name)
        country_cases = country_info[0]
        country_deaths = country_info[1]
        country_recoveries = country_info[2]
        
        country_daily_increase = daily_increase(country_cases)
        country_daily_death = daily_increase(country_deaths)
        country_daily_recovery = daily_increase(country_recoveries)
        
        country_plot(adjusted_dates, country_cases, country_daily_increase, country_daily_death, country_daily_recovery, country_name)

    countries = ['Albania',
    'Algeria',
    'Andorra',
    'Angola',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bhutan',
    'Bolivia',
    'Bosnia and Herzegovina',
    'Botswana',
    'Brazil',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burma',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Central African Republic',
    'Chad',
    'Chile',
    'Colombia',
    'Comoros',
    'Congo (Brazzaville)',
    'Congo (Kinshasa)',
    'Costa Rica',
    'Croatia',
    'Cuba',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Diamond Princess',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea',
    'Estonia',
    'Eswatini',
    'Ethiopia',
    'Fiji',
    'Finland',
    'France',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Greece',
    'Grenada',
    'Guatemala',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Holy See',
    'Honduras',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Ireland',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Korea, South',
    'Kosovo',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'MS Zaandam',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Mauritania',
    'Mauritius',
    'Mexico',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Mozambique',
    'Namibia',
    'Nepal',
    'Netherlands',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'North Macedonia',
    'Norway',
    'Oman',
    'Pakistan',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Qatar',
    'Romania',
    'Russia',
    'Rwanda',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'San Marino',
    'Sao Tome and Principe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Slovakia',
    'Slovenia',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'Sudan',
    'Suriname',
    'Sweden',
    'Switzerland',
    'Syria',
    'Taiwan*',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Trinidad and Tobago',
    'Tunisia',
    'Turkey',
    'US',
    'Uganda',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'Uruguay',
    'Uzbekistan',
    'Venezuela',
    'Vietnam',
    'West Bank and Gaza',
    'Western Sahara',
    'Yemen',
    'Zambia',
    'Zimbabwe',
    ]

    def plot_predictions(x, y, pred, algo_name, color):
        plt.style.use(['dark_background'])
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.title('COVID-19', size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('Cantidad de Casos', size=30)
        plt.legend(['Casos Confirmados', algo_name], prop={'size': 20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        nombre_archivo = " predicciones_covid_" + algo_name  + '.png'
        plt.savefig(nombre_archivo)
        #plt.show()

    if (prediccion_escoger == "RedesNeuronales"):
        plot_predictions(adjusted_dates, world_cases, neuronal_pred , 'Predicciones de Redes Neuronales a nivel general sobre la recuperacion', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Redes Neuronales a nivel general sobre la recuperacion' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "Lasso"):
        plot_predictions(adjusted_dates, world_cases, lasso_pred, 'Predicciones de regresión de Lasso a nivel general sobre la recuperacion', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de Lasso a nivel general sobre la recuperacion' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "RegresionCrestaBayesiana"):
        plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana sobre la recuperacion', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana sobre la recuperacion' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7


def predicciones_por_pais_mortalidad(prediccion_escoger, pais):
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    confirmado_por_pais = confirmed_df['Country/Region']==pais
    confirmed_df = confirmed_df[confirmado_por_pais]
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    muertos_por_pais = deaths_df['Country/Region']==pais
    deaths_df = deaths_df[muertos_por_pais]
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    recuperados_por_pais  = recoveries_df['Country/Region']==pais
    recoveries_df = recoveries_df[recuperados_por_pais]
    from datetime import datetime, timedelta
    d = datetime.today() - timedelta(days=1)
    fecha_actual = d.strftime('%m-%d-%Y')
    print(fecha_actual)
    fecha_actual2 = d.strftime('%Y-%m-%d')
    print(fecha_actual2)
    latest_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/" + fecha_actual + ".csv")
    us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+ fecha_actual + ".csv")   
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []
    total_deaths = [] 
    mortality_rate = []
    recovery_rate = [] 
    total_recovered = [] 
    total_active = [] 

    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()
        
        # confirmed, deaths, recovered, and active
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        total_recovered.append(recovered_sum)
        total_active.append(confirmed_sum-death_sum-recovered_sum)
        
        # calculate rates
        mortality_rate.append(death_sum/confirmed_sum)
        recovery_rate.append(recovered_sum/confirmed_sum)
    def daily_increase(data):
        d = [] 
        for i in range(len(data)):
            if i == 0:
                d.append(data[0])
            else:
                d.append(data[i]-data[i-1])
        return d 

    def moving_average(data, window_size):
        moving_average = []
        for i in range(len(data)):
            if i + window_size < len(data):
                moving_average.append(np.mean(data[i:i+window_size]))
            else:
                moving_average.append(np.mean(data[i:len(data)]))
        return moving_average

    window = 7

    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg= moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)

    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)

    world_daily_recovery = daily_increase(total_recovered)
    world_recovery_avg = moving_average(total_recovered, window)
    world_daily_recovery_avg = moving_average(world_daily_recovery, window)

    world_active_avg = moving_average(total_active, window)
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    total_recovered = np.array(total_recovered).reshape(-1, 1)
    days_in_future = 50
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-50]

    import datetime
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.05, shuffle=False)

    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    #plt.plot(y_test_confirmed)
    #plt.plot(svm_test_pred)
    #plt.legend(['Test Data', 'SVM Predictions'])

    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
    bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)

    #plt.plot(y_test_confirmed)
    #plt.plot(test_linear_pred)
    #plt.legend(['Test Data', 'Polynomial Regression Predictions'])

    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                    'normalize' : normalize}

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
    bayesian_search.best_params_
    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    #plt.plot(y_test_confirmed)
    #plt.plot(test_bayesian_pred)
    #plt.legend(['Test Data', 'Predicciones de Cresta Bayesianas'])

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(bayesian_search, y_test_confirmed, test_bayesian_pred, cv=4)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*-1, scores.std() * 2))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train_confirmed, y_train_confirmed)
    print(clf.coef_)
    print(clf.intercept_)
    testlassopredict = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])


    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, y_train_confirmed, testlassopredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    lasso_pred = clf.predict(future_forcast)

    from sklearn.linear_model import LassoCV
    from sklearn.datasets import make_regression
    reg = LassoCV(cv=5, random_state=0).fit(X_train_confirmed, y_train_confirmed)
    #reg.score(X, y)
    testlassoCVpredict = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])

    scores = cross_val_score(reg, X_train_confirmed, testlassoCVpredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.linear_model import ElasticNetCV
    from sklearn.datasets import make_regression
    regr = ElasticNetCV(cv=50, random_state=0)
    regr.fit(X_train_confirmed, y_train_confirmed)
    print(regr.alpha_)
    print(regr.intercept_)
    test_elasticnet = regr.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(test_elasticnet)
    #plt.legend(['Original Data', 'ElasticNetCV Predictions'])

    from sklearn import linear_model
    reg = linear_model.BayesianRidge()
    reg.fit(X_train_confirmed, y_train_confirmed)
    pred_bayesian_ridge = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Bayesian Ridge Predictions'])


    scores = cross_val_score(regr, X_train_confirmed, test_elasticnet, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000000000000)
    clf.fit(X_train_confirmed, y_train_confirmed)
    predictneural = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(predictneural)
    #plt.legend(['Original Data', 'Neural Network Predictions'])

    neuronal_pred = clf.predict(future_forcast)


    scores = cross_val_score(clf, X_train_confirmed, predictneural, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    def country_plot(x, y1, y2, y3, y4, country):
        # window is set as 14 in in the beginning of the notebook 
        confirmed_avg = moving_average(y1, window)
        confirmed_increase_avg = moving_average(y2, window)
        death_increase_avg = moving_average(y3, window)
        recovery_increase_avg = moving_average(y4, window)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure(figsize=(16, 10))
        plt.plot(x, y1)
        plt.plot(x, confirmed_avg, color='red', linestyle='dashed')
        plt.legend(['{} Confirmed Cases'.format(country), 'Moving Average {} Days'.format(window)], prop={'size': 20})
        plt.title('{} Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y2)
        plt.plot(x, confirmed_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Cases'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y3)
        plt.plot(x, death_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Deaths'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Deaths'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y4)
        plt.plot(x, recovery_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Recoveries'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        
    # helper function for getting country's cases, deaths, and recoveries        
    def get_country_info(country_name):
        country_cases = []
        country_deaths = []
        country_recoveries = []  
        
        for i in dates:
            country_cases.append(confirmed_df[confirmed_df['Country/Region']==country_name][i].sum())
            country_deaths.append(deaths_df[deaths_df['Country/Region']==country_name][i].sum())
            country_recoveries.append(recoveries_df[recoveries_df['Country/Region']==country_name][i].sum())
        return (country_cases, country_deaths, country_recoveries)
        
        
    def country_visualizations(country_name):
        country_info = get_country_info(country_name)
        country_cases = country_info[0]
        country_deaths = country_info[1]
        country_recoveries = country_info[2]
        
        country_daily_increase = daily_increase(country_cases)
        country_daily_death = daily_increase(country_deaths)
        country_daily_recovery = daily_increase(country_recoveries)
        
        country_plot(adjusted_dates, country_cases, country_daily_increase, country_daily_death, country_daily_recovery, country_name)

    countries = ['Albania',
    'Algeria',
    'Andorra',
    'Angola',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bhutan',
    'Bolivia',
    'Bosnia and Herzegovina',
    'Botswana',
    'Brazil',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burma',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Central African Republic',
    'Chad',
    'Chile',
    'Colombia',
    'Comoros',
    'Congo (Brazzaville)',
    'Congo (Kinshasa)',
    'Costa Rica',
    'Croatia',
    'Cuba',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Diamond Princess',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea',
    'Estonia',
    'Eswatini',
    'Ethiopia',
    'Fiji',
    'Finland',
    'France',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Greece',
    'Grenada',
    'Guatemala',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Holy See',
    'Honduras',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Ireland',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Korea, South',
    'Kosovo',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'MS Zaandam',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Mauritania',
    'Mauritius',
    'Mexico',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Mozambique',
    'Namibia',
    'Nepal',
    'Netherlands',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'North Macedonia',
    'Norway',
    'Oman',
    'Pakistan',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Qatar',
    'Romania',
    'Russia',
    'Rwanda',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'San Marino',
    'Sao Tome and Principe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Slovakia',
    'Slovenia',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'Sudan',
    'Suriname',
    'Sweden',
    'Switzerland',
    'Syria',
    'Taiwan*',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Trinidad and Tobago',
    'Tunisia',
    'Turkey',
    'US',
    'Uganda',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'Uruguay',
    'Uzbekistan',
    'Venezuela',
    'Vietnam',
    'West Bank and Gaza',
    'Western Sahara',
    'Yemen',
    'Zambia',
    'Zimbabwe',
    ]

    def plot_predictions(x, y, pred, algo_name, color):
        plt.style.use(['dark_background'])
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.title('COVID-19', size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('Cantidad de Casos', size=30)
        plt.legend(['Casos Confirmados', algo_name], prop={'size': 20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        nombre_archivo = " predicciones_covid_" + algo_name  + '.png'
        plt.savefig(nombre_archivo)
        #plt.show()

    if (prediccion_escoger == "RedesNeuronales"):
        plot_predictions(adjusted_dates, world_cases, neuronal_pred , 'Predicciones de Redes Neuronales en ' + pais + ' en base a la tasa de mortalidad', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Redes Neuronales en ' + pais+ ' en base a la tasa de mortalidad' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "Lasso"):
        plot_predictions(adjusted_dates, world_cases, lasso_pred, 'Predicciones de Lasso en ' + pais+ ' en base a la tasa de mortalidad', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Lasso en ' + pais+ ' en base a la tasa de mortalidad' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "RegresionCrestaBayesiana"):
        plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana en ' + pais+ ' en base a la tasa de mortalidad', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana en '+ pais + ' en base a la tasa de mortalidad'+ '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7

def predicciones_por_pais(prediccion_escoger, pais):
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    confirmado_por_pais = confirmed_df['Country/Region']==pais
    confirmed_df = confirmed_df[confirmado_por_pais]
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    muertos_por_pais = deaths_df['Country/Region']==pais
    deaths_df = deaths_df[muertos_por_pais]
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    recuperados_por_pais  = recoveries_df['Country/Region']==pais
    recoveries_df = recoveries_df[recuperados_por_pais]
    from datetime import datetime, timedelta
    d = datetime.today() - timedelta(days=1)
    fecha_actual = d.strftime('%m-%d-%Y')
    print(fecha_actual)
    fecha_actual2 = d.strftime('%Y-%m-%d')
    print(fecha_actual2)
    latest_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/" + fecha_actual + ".csv")
    us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+ fecha_actual + ".csv")   
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []
    total_deaths = [] 
    mortality_rate = []
    recovery_rate = [] 
    total_recovered = [] 
    total_active = [] 

    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()
        
        # confirmed, deaths, recovered, and active
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        total_recovered.append(recovered_sum)
        total_active.append(confirmed_sum-death_sum-recovered_sum)
        
        # calculate rates
        mortality_rate.append(death_sum/confirmed_sum)
        recovery_rate.append(recovered_sum/confirmed_sum)
    def daily_increase(data):
        d = [] 
        for i in range(len(data)):
            if i == 0:
                d.append(data[0])
            else:
                d.append(data[i]-data[i-1])
        return d 

    def moving_average(data, window_size):
        moving_average = []
        for i in range(len(data)):
            if i + window_size < len(data):
                moving_average.append(np.mean(data[i:i+window_size]))
            else:
                moving_average.append(np.mean(data[i:len(data)]))
        return moving_average

    window = 7

    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg= moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)

    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)

    world_daily_recovery = daily_increase(total_recovered)
    world_recovery_avg = moving_average(total_recovered, window)
    world_daily_recovery_avg = moving_average(world_daily_recovery, window)

    world_active_avg = moving_average(total_active, window)
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    total_recovered = np.array(total_recovered).reshape(-1, 1)
    days_in_future = 50
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-50]

    import datetime
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.05, shuffle=False)

    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    #plt.plot(y_test_confirmed)
    #plt.plot(svm_test_pred)
    #plt.legend(['Test Data', 'SVM Predictions'])

    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
    bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)

    #plt.plot(y_test_confirmed)
    #plt.plot(test_linear_pred)
    #plt.legend(['Test Data', 'Polynomial Regression Predictions'])

    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                    'normalize' : normalize}

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
    bayesian_search.best_params_
    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    #plt.plot(y_test_confirmed)
    #plt.plot(test_bayesian_pred)
    #plt.legend(['Test Data', 'Predicciones de Cresta Bayesianas'])

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(bayesian_search, y_test_confirmed, test_bayesian_pred, cv=4)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*-1, scores.std() * 2))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train_confirmed, y_train_confirmed)
    print(clf.coef_)
    print(clf.intercept_)
    testlassopredict = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])


    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, y_train_confirmed, testlassopredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    lasso_pred = clf.predict(future_forcast)

    from sklearn.linear_model import LassoCV
    from sklearn.datasets import make_regression
    reg = LassoCV(cv=5, random_state=0).fit(X_train_confirmed, y_train_confirmed)
    #reg.score(X, y)
    testlassoCVpredict = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])

    scores = cross_val_score(reg, X_train_confirmed, testlassoCVpredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.linear_model import ElasticNetCV
    from sklearn.datasets import make_regression
    regr = ElasticNetCV(cv=50, random_state=0)
    regr.fit(X_train_confirmed, y_train_confirmed)
    print(regr.alpha_)
    print(regr.intercept_)
    test_elasticnet = regr.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(test_elasticnet)
    #plt.legend(['Original Data', 'ElasticNetCV Predictions'])

    from sklearn import linear_model
    reg = linear_model.BayesianRidge()
    reg.fit(X_train_confirmed, y_train_confirmed)
    pred_bayesian_ridge = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Bayesian Ridge Predictions'])


    scores = cross_val_score(regr, X_train_confirmed, test_elasticnet, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000000000000)
    clf.fit(X_train_confirmed, y_train_confirmed)
    predictneural = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(predictneural)
    #plt.legend(['Original Data', 'Neural Network Predictions'])

    neuronal_pred = clf.predict(future_forcast)


    scores = cross_val_score(clf, X_train_confirmed, predictneural, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    def country_plot(x, y1, y2, y3, y4, country):
        # window is set as 14 in in the beginning of the notebook 
        confirmed_avg = moving_average(y1, window)
        confirmed_increase_avg = moving_average(y2, window)
        death_increase_avg = moving_average(y3, window)
        recovery_increase_avg = moving_average(y4, window)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure(figsize=(16, 10))
        plt.plot(x, y1)
        plt.plot(x, confirmed_avg, color='red', linestyle='dashed')
        plt.legend(['{} Confirmed Cases'.format(country), 'Moving Average {} Days'.format(window)], prop={'size': 20})
        plt.title('{} Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y2)
        plt.plot(x, confirmed_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Cases'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y3)
        plt.plot(x, death_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Deaths'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Deaths'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y4)
        plt.plot(x, recovery_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Recoveries'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        
    # helper function for getting country's cases, deaths, and recoveries        
    def get_country_info(country_name):
        country_cases = []
        country_deaths = []
        country_recoveries = []  
        
        for i in dates:
            country_cases.append(confirmed_df[confirmed_df['Country/Region']==country_name][i].sum())
            country_deaths.append(deaths_df[deaths_df['Country/Region']==country_name][i].sum())
            country_recoveries.append(recoveries_df[recoveries_df['Country/Region']==country_name][i].sum())
        return (country_cases, country_deaths, country_recoveries)
        
        
    def country_visualizations(country_name):
        country_info = get_country_info(country_name)
        country_cases = country_info[0]
        country_deaths = country_info[1]
        country_recoveries = country_info[2]
        
        country_daily_increase = daily_increase(country_cases)
        country_daily_death = daily_increase(country_deaths)
        country_daily_recovery = daily_increase(country_recoveries)
        
        country_plot(adjusted_dates, country_cases, country_daily_increase, country_daily_death, country_daily_recovery, country_name)

    countries = ['Albania',
    'Algeria',
    'Andorra',
    'Angola',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bhutan',
    'Bolivia',
    'Bosnia and Herzegovina',
    'Botswana',
    'Brazil',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burma',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Central African Republic',
    'Chad',
    'Chile',
    'Colombia',
    'Comoros',
    'Congo (Brazzaville)',
    'Congo (Kinshasa)',
    'Costa Rica',
    'Croatia',
    'Cuba',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Diamond Princess',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea',
    'Estonia',
    'Eswatini',
    'Ethiopia',
    'Fiji',
    'Finland',
    'France',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Greece',
    'Grenada',
    'Guatemala',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Holy See',
    'Honduras',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Ireland',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Korea, South',
    'Kosovo',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'MS Zaandam',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Mauritania',
    'Mauritius',
    'Mexico',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Mozambique',
    'Namibia',
    'Nepal',
    'Netherlands',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'North Macedonia',
    'Norway',
    'Oman',
    'Pakistan',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Qatar',
    'Romania',
    'Russia',
    'Rwanda',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'San Marino',
    'Sao Tome and Principe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Slovakia',
    'Slovenia',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'Sudan',
    'Suriname',
    'Sweden',
    'Switzerland',
    'Syria',
    'Taiwan*',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Trinidad and Tobago',
    'Tunisia',
    'Turkey',
    'US',
    'Uganda',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'Uruguay',
    'Uzbekistan',
    'Venezuela',
    'Vietnam',
    'West Bank and Gaza',
    'Western Sahara',
    'Yemen',
    'Zambia',
    'Zimbabwe',
    ]

    def plot_predictions(x, y, pred, algo_name, color):
        plt.style.use(['dark_background'])
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.title('COVID-19', size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('Cantidad de Casos', size=30)
        plt.legend(['Casos Confirmados', algo_name], prop={'size': 20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        nombre_archivo = " predicciones_covid_" + algo_name  + '.png'
        plt.savefig(nombre_archivo)
        #plt.show()

    if (prediccion_escoger == "RedesNeuronales"):
        plot_predictions(adjusted_dates, world_cases, neuronal_pred , 'Predicciones de Redes Neuronales en ' + pais , 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Redes Neuronales en ' + pais + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "Lasso"):
        plot_predictions(adjusted_dates, world_cases, lasso_pred, 'Predicciones de Lasso en ' + pais, 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Lasso en ' + pais  + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "RegresionCrestaBayesiana"):
        plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana en ' + pais , 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana en '+ pais  + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7

def predicciones_por_pais_recuperacion(prediccion_escoger, pais):
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    confirmado_por_pais = confirmed_df['Country/Region']==pais
    confirmed_df = confirmed_df[confirmado_por_pais]
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    muertos_por_pais = deaths_df['Country/Region']==pais
    deaths_df = deaths_df[muertos_por_pais]
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    recuperados_por_pais  = recoveries_df['Country/Region']==pais
    recoveries_df = recoveries_df[recuperados_por_pais]
    from datetime import datetime, timedelta
    d = datetime.today() - timedelta(days=1)
    fecha_actual = d.strftime('%m-%d-%Y')
    print(fecha_actual)
    fecha_actual2 = d.strftime('%Y-%m-%d')
    print(fecha_actual2)
    latest_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/" + fecha_actual + ".csv")
    us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+ fecha_actual + ".csv")   
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []
    total_deaths = [] 
    mortality_rate = []
    recovery_rate = [] 
    total_recovered = [] 
    total_active = [] 

    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()
        
        # confirmed, deaths, recovered, and active
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        total_recovered.append(recovered_sum)
        total_active.append(confirmed_sum-death_sum-recovered_sum)
        
        # calculate rates
        mortality_rate.append(death_sum/confirmed_sum)
        recovery_rate.append(recovered_sum/confirmed_sum)
    def daily_increase(data):
        d = [] 
        for i in range(len(data)):
            if i == 0:
                d.append(data[0])
            else:
                d.append(data[i]-data[i-1])
        return d 

    def moving_average(data, window_size):
        moving_average = []
        for i in range(len(data)):
            if i + window_size < len(data):
                moving_average.append(np.mean(data[i:i+window_size]))
            else:
                moving_average.append(np.mean(data[i:len(data)]))
        return moving_average

    window = 7

    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg= moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)

    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)

    world_daily_recovery = daily_increase(total_recovered)
    world_recovery_avg = moving_average(total_recovered, window)
    world_daily_recovery_avg = moving_average(world_daily_recovery, window)

    world_active_avg = moving_average(total_active, window)
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    total_recovered = np.array(total_recovered).reshape(-1, 1)
    days_in_future = 50
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-50]

    import datetime
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.05, shuffle=False)

    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    #plt.plot(y_test_confirmed)
    #plt.plot(svm_test_pred)
    #plt.legend(['Test Data', 'SVM Predictions'])

    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
    bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)

    #plt.plot(y_test_confirmed)
    #plt.plot(test_linear_pred)
    #plt.legend(['Test Data', 'Polynomial Regression Predictions'])

    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                    'normalize' : normalize}

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
    bayesian_search.best_params_
    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    #plt.plot(y_test_confirmed)
    #plt.plot(test_bayesian_pred)
    #plt.legend(['Test Data', 'Predicciones de Cresta Bayesianas'])

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(bayesian_search, y_test_confirmed, test_bayesian_pred, cv=4)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*-1, scores.std() * 2))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train_confirmed, y_train_confirmed)
    print(clf.coef_)
    print(clf.intercept_)
    testlassopredict = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])


    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, y_train_confirmed, testlassopredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    lasso_pred = clf.predict(future_forcast)

    from sklearn.linear_model import LassoCV
    from sklearn.datasets import make_regression
    reg = LassoCV(cv=5, random_state=0).fit(X_train_confirmed, y_train_confirmed)
    #reg.score(X, y)
    testlassoCVpredict = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Lasso Predictions'])

    scores = cross_val_score(reg, X_train_confirmed, testlassoCVpredict, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.linear_model import ElasticNetCV
    from sklearn.datasets import make_regression
    regr = ElasticNetCV(cv=50, random_state=0)
    regr.fit(X_train_confirmed, y_train_confirmed)
    print(regr.alpha_)
    print(regr.intercept_)
    test_elasticnet = regr.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(test_elasticnet)
    #plt.legend(['Original Data', 'ElasticNetCV Predictions'])

    from sklearn import linear_model
    reg = linear_model.BayesianRidge()
    reg.fit(X_train_confirmed, y_train_confirmed)
    pred_bayesian_ridge = reg.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(testlassopredict)
    #plt.legend(['Original Data', 'Bayesian Ridge Predictions'])


    scores = cross_val_score(regr, X_train_confirmed, test_elasticnet, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000000000000)
    clf.fit(X_train_confirmed, y_train_confirmed)
    predictneural = clf.predict(X_train_confirmed)
    #plt.plot(y_train_confirmed)
    #plt.plot(predictneural)
    #plt.legend(['Original Data', 'Neural Network Predictions'])

    neuronal_pred = clf.predict(future_forcast)


    scores = cross_val_score(clf, X_train_confirmed, predictneural, cv=5)
    scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    def country_plot(x, y1, y2, y3, y4, country):
        # window is set as 14 in in the beginning of the notebook 
        confirmed_avg = moving_average(y1, window)
        confirmed_increase_avg = moving_average(y2, window)
        death_increase_avg = moving_average(y3, window)
        recovery_increase_avg = moving_average(y4, window)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure(figsize=(16, 10))
        plt.plot(x, y1)
        plt.plot(x, confirmed_avg, color='red', linestyle='dashed')
        plt.legend(['{} Confirmed Cases'.format(country), 'Moving Average {} Days'.format(window)], prop={'size': 20})
        plt.title('{} Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y2)
        plt.plot(x, confirmed_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Cases'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y3)
        plt.plot(x, death_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Deaths'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Deaths'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.style.use(['dark_background'])
        plt.rcParams['axes.facecolor'] = 'black'

        plt.figure(figsize=(16, 10))
        plt.bar(x, y4)
        plt.plot(x, recovery_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(window), '{} Daily Increase in Confirmed Recoveries'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        
    # helper function for getting country's cases, deaths, and recoveries        
    def get_country_info(country_name):
        country_cases = []
        country_deaths = []
        country_recoveries = []  
        
        for i in dates:
            country_cases.append(confirmed_df[confirmed_df['Country/Region']==country_name][i].sum())
            country_deaths.append(deaths_df[deaths_df['Country/Region']==country_name][i].sum())
            country_recoveries.append(recoveries_df[recoveries_df['Country/Region']==country_name][i].sum())
        return (country_cases, country_deaths, country_recoveries)
        
        
    def country_visualizations(country_name):
        country_info = get_country_info(country_name)
        country_cases = country_info[0]
        country_deaths = country_info[1]
        country_recoveries = country_info[2]
        
        country_daily_increase = daily_increase(country_cases)
        country_daily_death = daily_increase(country_deaths)
        country_daily_recovery = daily_increase(country_recoveries)
        
        country_plot(adjusted_dates, country_cases, country_daily_increase, country_daily_death, country_daily_recovery, country_name)

    countries = ['Albania',
    'Algeria',
    'Andorra',
    'Angola',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bhutan',
    'Bolivia',
    'Bosnia and Herzegovina',
    'Botswana',
    'Brazil',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burma',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Central African Republic',
    'Chad',
    'Chile',
    'Colombia',
    'Comoros',
    'Congo (Brazzaville)',
    'Congo (Kinshasa)',
    'Costa Rica',
    'Croatia',
    'Cuba',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Diamond Princess',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea',
    'Estonia',
    'Eswatini',
    'Ethiopia',
    'Fiji',
    'Finland',
    'France',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Greece',
    'Grenada',
    'Guatemala',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Holy See',
    'Honduras',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Ireland',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Korea, South',
    'Kosovo',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'MS Zaandam',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Mauritania',
    'Mauritius',
    'Mexico',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Mozambique',
    'Namibia',
    'Nepal',
    'Netherlands',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'North Macedonia',
    'Norway',
    'Oman',
    'Pakistan',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Qatar',
    'Romania',
    'Russia',
    'Rwanda',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'San Marino',
    'Sao Tome and Principe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Slovakia',
    'Slovenia',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'Sudan',
    'Suriname',
    'Sweden',
    'Switzerland',
    'Syria',
    'Taiwan*',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Trinidad and Tobago',
    'Tunisia',
    'Turkey',
    'US',
    'Uganda',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'Uruguay',
    'Uzbekistan',
    'Venezuela',
    'Vietnam',
    'West Bank and Gaza',
    'Western Sahara',
    'Yemen',
    'Zambia',
    'Zimbabwe',
    ]

    def plot_predictions(x, y, pred, algo_name, color):
        plt.style.use(['dark_background'])
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.plot(future_forcast, pred, linestyle='dashed', color=color)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.title('COVID-19', size=30)
        plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
        plt.ylabel('Cantidad de Casos', size=30)
        plt.legend(['Casos Confirmados', algo_name], prop={'size': 20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        nombre_archivo = " predicciones_covid_" + algo_name  + '.png'
        plt.savefig(nombre_archivo)
        #plt.show()

    if (prediccion_escoger == "RedesNeuronales"):
        plot_predictions(adjusted_dates, world_cases, neuronal_pred , 'Predicciones de Redes Neuronales en ' + pais + ' en base a la tasa de recuperacion', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Redes Neuronales en ' + pais+ ' en base a la tasa de recuperacion' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "Lasso"):
        plot_predictions(adjusted_dates, world_cases, lasso_pred, 'Predicciones de Lasso en ' + pais + ' en base a la tasa de recuperacion', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de Lasso en ' + pais + ' en base a la tasa de recuperacion' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7
    if (prediccion_escoger == "RegresionCrestaBayesiana"):
        plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana en ' + pais + ' en base a la tasa de recuperacion', 'orange')
        nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana en '+ pais + ' en base a la tasa de recuperacion' + '.png'
        with open(nombre_archivo, "rb") as image_file:
            encoded_string_covid7 = base64.b64encode(image_file.read())
            print(encoded_string_covid7)
        return encoded_string_covid7



#cluster_covid()
#predicciones_por_pais_mortalidad("RedesNeuronales","Guatemala")
#predicciones_por_pais_mortalidad("RegresionCrestaBayesiana","Guatemala")
#predicciones_por_pais_mortalidad("Lasso","Guatemala")
#predicciones_por_pais_mortalidad("RedesNeuronales","Canada")
#predicciones_por_pais_mortalidad("RegresionCrestaBayesiana","Canada")
#predicciones_por_pais_mortalidad("Lasso","Canada")

#predicciones_por_pais("RedesNeuronales","Canada")
#predicciones_por_pais("RegresionCrestaBayesiana","Canada")
#predicciones_por_pais("Lasso","Canada")
#predicciones_por_pais("RedesNeuronales","Guatemala")
#predicciones_por_pais("RegresionCrestaBayesiana","Guatemala")
#predicciones_por_pais("Lasso","Guatemala")

#predicciones_por_pais_recuperacion("RedesNeuronales","Canada")
#predicciones_por_pais_recuperacion("RegresionCrestaBayesiana","Canada")
#predicciones_por_pais_recuperacion("Lasso","Canada")
#predicciones_por_pais_recuperacion("RedesNeuronales","Guatemala")
#predicciones_por_pais_recuperacion("RegresionCrestaBayesiana","Guatemala")
#predicciones_por_pais_recuperacion("Lasso","Guatemala")
#predicciones_mortalidad_generales("RedesNeuronales")
#predicciones_mortalidad_generales("Lasso")
#predicciones_mortalidad_generales("RegresionCrestaBayesiana")

#predicciones_generales("RedesNeuronales")
#predicciones_generales("Lasso")
#predicciones_generales("RegresionCrestaBayesiana")

#predicciones_recuperacion_generales("RedesNeuronales")
#predicciones_recuperacion_generales("Lasso")
#predicciones_recuperacion_generales("RegresionCrestaBayesiana")
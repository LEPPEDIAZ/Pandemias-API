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
from sklearn.model_selection import cross_val_score
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
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
days_in_future =  150
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:- 150]
import datetime
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[ 150:], world_cases[ 150:], test_size=0.05, shuffle=False) 
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(y_test_confirmed)
plt.plot(svm_test_pred)
plt.legend(['Test Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))

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
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])

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
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])

def country_plot(x, y1, y2, y3, y4, country):
    # window is set as 14 in in the beginning of the notebook 
    confirmed_avg = moving_average(y1, window)
    confirmed_increase_avg = moving_average(y2, window)
    death_increase_avg = moving_average(y3, window)
    recovery_increase_avg = moving_average(y4, window)
    
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
    plt.figure(figsize=(16, 10))
    plt.plot(x, y)
    plt.plot(future_forcast, pred, linestyle='dashed', color=color)
    plt.title('COVID-19', size=30)
    plt.xlabel('Iniciando desde el dia 1/22/2020', size=30)
    plt.ylabel('Cantidad de Casos', size=30)
    plt.legend(['Casos', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    nombre_archivo = " predicciones_covid_" + algo_name  + '.png'
    plt.savefig(nombre_archivo)
    ##plt.show()
    
#plot_predictions(adjusted_dates, world_cases, svm_pred, 'Predicciones de SVM de Confirmados de COVID-19', 'purple')
plot_predictions(adjusted_dates, world_cases, linear_pred, 'Predicciones de Regresion Polinomeal de Confirmados de COVID-19', 'orange')
nombre_archivo = " predicciones_covid_" + 'Predicciones de Regresion Polinomeal de Confirmados de COVID-19'  + '.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_covid1 = base64.b64encode(image_file.read())
    print(encoded_string_covid1)
plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana de Confirmados de COVID-19', 'green')
nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana de Confirmados de COVID-19'  + '.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_covid2 = base64.b64encode(image_file.read())
    print(encoded_string_covid2)
X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(days_since_1_22[ 150:], total_deaths[ 150:], test_size=0.05, shuffle=False)
svm_death = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_death.fit(X_train_death, y_train_death)
svm_pred_death = svm_death.predict(future_forcast)
svm_test_pred_death = svm_death.predict(X_test_death)
plt.plot(y_test_death)
plt.plot(svm_test_pred_death)
plt.legend(['Test Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svm_test_pred_death, y_test_death))
print('MSE:',mean_squared_error(svm_test_pred_death, y_test_death))

#plot_predictions(adjusted_dates, total_deaths, svm_pred_death, 'SVM Predictions', 'purple')
poly = PolynomialFeatures(degree=5)
poly_X_train_death = poly.fit_transform(X_train_death)
poly_X_test_death = poly.fit_transform(X_test_death)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=5)
bayesian_poly_X_train_death = bayesian_poly.fit_transform(X_train_death)
bayesian_poly_X_test_death = bayesian_poly.fit_transform(X_test_death)
bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_death, y_train_death)
test_linear_pred = linear_model.predict(poly_X_test_death)
linear_pred_death = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_death))
print('MSE:',mean_squared_error(test_linear_pred, y_test_death))


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
bayesian_search.fit(bayesian_poly_X_train_death, y_train_death)
bayesian_search.best_params_
bayesian_death = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_death.predict(bayesian_poly_X_test_death)
bayesian_pred = bayesian_death.predict(bayesian_poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_death))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_death))

plot_predictions(adjusted_dates, total_deaths, linear_pred_death, 'Predicciones de Regresion Polinomeal de Mortalidad de COVID-19', 'purple')
nombre_archivo = " predicciones_covid_" + 'Predicciones de Regresion Polinomeal de Mortalidad de COVID-19'  + '.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_covid3 = base64.b64encode(image_file.read())
    print(encoded_string_covid3)

plot_predictions(adjusted_dates, total_deaths, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana de Mortalidad de COVID-19', 'purple')
nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana de Mortalidad de COVID-19'  + '.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_covid4 = base64.b64encode(image_file.read())
    print(encoded_string_covid4)


X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(days_since_1_22[ 150:], total_recovered[ 150:], test_size=0.05, shuffle=False)
poly = PolynomialFeatures(degree=5)
poly_X_train_recovered = poly.fit_transform(X_train_recovered)
poly_X_test_recovered = poly.fit_transform(X_test_recovered)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=5)
bayesian_poly_X_train_recovered = bayesian_poly.fit_transform(X_train_recovered)
bayesian_poly_X_test_recovered = bayesian_poly.fit_transform(X_test_recovered)
bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_recovered, y_train_recovered)
test_linear_pred = linear_model.predict(poly_X_test_recovered)
linear_pred_recovered = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_recovered))
print('MSE:',mean_squared_error(test_linear_pred, y_test_recovered))

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
bayesian_search.fit(bayesian_poly_X_train_recovered, y_train_recovered)
bayesian_search.best_params_
bayesian_recovered = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_recovered.predict(bayesian_poly_X_test_recovered)
bayesian_pred = bayesian_recovered.predict(bayesian_poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_recovered))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_recovered))

plot_predictions(adjusted_dates, total_deaths, linear_pred_death, 'Predicciones de Regresion Polinomeal de Recuperados de COVID-19', 'purple')
nombre_archivo = " predicciones_covid_" + 'Predicciones de Regresion Polinomeal de Recuperados de COVID-19'  + '.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_covid5 = base64.b64encode(image_file.read())
    print(encoded_string_covid5)

plot_predictions(adjusted_dates, total_deaths, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana de Recuperados de COVID-19', 'purple')
nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana de Recuperados de COVID-19'  + '.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_covid6 = base64.b64encode(image_file.read())
    print(encoded_string_covid6)


def prediccion_por_pais_confirmados_polinomeal(pais):
    confirmado_por_pais =  confirmed_df['Country/Region']==pais
    confirmado_por_pais = confirmed_df[confirmado_por_pais]
    confirmado_por_pais
    muertos_por_pais =  deaths_df['Country/Region']==pais
    muertos_por_pais = deaths_df[muertos_por_pais]
    muertos_por_pais
    recuperados_por_pais =  recoveries_df['Country/Region']==pais
    recuperados_por_pais = recoveries_df[recuperados_por_pais]
    recuperados_por_pais
    cols = confirmado_por_pais.keys()
    confirmed = confirmado_por_pais.loc[:, cols[4]:cols[-1]]
    deaths = muertos_por_pais.loc[:, cols[4]:cols[-1]]
    recoveries = recuperados_por_pais.loc[:, cols[4]:cols[-1]]
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
    days_in_future =  150
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:- 150]
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[ 150:], world_cases[ 150:], test_size=0.05, shuffle=False)
    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    plt.plot(y_test_confirmed)
    plt.plot(svm_test_pred)
    plt.legend(['Test Data', 'SVM Predictions'])
    print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
    print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
    #plot_predictions(adjusted_dates, world_cases, svm_pred, 'Predicciones de SVM', 'purple')
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
    print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
    print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))

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
    print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
    print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))

    plot_predictions(adjusted_dates, world_cases, linear_pred, 'Predicciones de Regresion Polinomeal para ' + pais + ' sobre la  cantidad de Confirmados', 'orange')
    nombre_archivo = " predicciones_covid_" + 'Predicciones de Regresion Polinomeal para ' + pais + ' sobre la  cantidad de Confirmados' + '.png'
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid7 = base64.b64encode(image_file.read())
        print(encoded_string_covid7)
    return encoded_string_covid7

prediccion_por_pais_confirmados_polinomeal('Guatemala')

def prediccion_por_pais_cresta_bayesiana(pais):
    confirmado_por_pais =  confirmed_df['Country/Region']==pais
    confirmado_por_pais = confirmed_df[confirmado_por_pais]
    confirmado_por_pais
    muertos_por_pais =  deaths_df['Country/Region']==pais
    muertos_por_pais = deaths_df[muertos_por_pais]
    muertos_por_pais
    recuperados_por_pais =  recoveries_df['Country/Region']==pais
    recuperados_por_pais = recoveries_df[recuperados_por_pais]
    recuperados_por_pais
    cols = confirmado_por_pais.keys()
    confirmed = confirmado_por_pais.loc[:, cols[4]:cols[-1]]
    deaths = muertos_por_pais.loc[:, cols[4]:cols[-1]]
    recoveries = recuperados_por_pais.loc[:, cols[4]:cols[-1]]
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
    days_in_future =  150
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:- 150]
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[ 150:], world_cases[ 150:], test_size=0.05, shuffle=False)
    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    plt.plot(y_test_confirmed)
    plt.plot(svm_test_pred)
    plt.legend(['Test Data', 'SVM Predictions'])
    print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
    print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
    #plot_predictions(adjusted_dates, world_cases, svm_pred, 'Predicciones de SVM', 'purple')
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
    print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
    print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))

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
    print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
    print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
    plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana para ' + pais + ' sobre la cantidad de Confirmados', 'green')
    nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana para ' + pais + ' sobre la cantidad de Confirmados' + '.png'
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid8 = base64.b64encode(image_file.read())
        print(encoded_string_covid8)
    return encoded_string_covid8
    
prediccion_por_pais_cresta_bayesiana('Guatemala')

def prediccion_por_pais_mortalidad_polinomeal(pais):
    X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(days_since_1_22[ 150:], total_deaths[ 150:], test_size=0.05, shuffle=False)
    svm_death = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_death.fit(X_train_death, y_train_death)
    svm_pred_death = svm_death.predict(future_forcast)
    svm_test_pred_death = svm_death.predict(X_test_death)
    plt.plot(y_test_death)
    plt.plot(svm_test_pred_death)
    plt.legend(['Test Data', 'SVM Predictions'])
    print('MAE:', mean_absolute_error(svm_test_pred_death, y_test_death))
    print('MSE:',mean_squared_error(svm_test_pred_death, y_test_death))

    poly = PolynomialFeatures(degree=5)
    poly_X_train_death = poly.fit_transform(X_train_death)
    poly_X_test_death = poly.fit_transform(X_test_death)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_death = bayesian_poly.fit_transform(X_train_death)
    bayesian_poly_X_test_death = bayesian_poly.fit_transform(X_test_death)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_death, y_train_death)
    test_linear_pred = linear_model.predict(poly_X_test_death)
    linear_pred_death = linear_model.predict(poly_future_forcast)
    print('MAE:', mean_absolute_error(test_linear_pred, y_test_death))
    print('MSE:',mean_squared_error(test_linear_pred, y_test_death))

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
    bayesian_search.fit(bayesian_poly_X_train_death, y_train_death)
    bayesian_search.best_params_
    bayesian_death = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_death.predict(bayesian_poly_X_test_death)
    bayesian_pred = bayesian_death.predict(bayesian_poly_future_forcast)
    print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_death))
    print('MSE:',mean_squared_error(test_bayesian_pred, y_test_death))

    #plot_predictions(adjusted_dates, total_deaths, svm_pred_death, 'SVM Predictions', 'purple')
    plot_predictions(adjusted_dates, total_deaths, linear_pred_death,'Predicciones de Regresion Polinomeal para ' + pais + ' sobre la mortalidad', 'purple')
    nombre_archivo = " predicciones_covid_" + 'Predicciones de Regresion Polinomeal para ' + pais + ' sobre la mortalidad' + '.png'
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid8 = base64.b64encode(image_file.read())
        print(encoded_string_covid8)
    return encoded_string_covid8
    
def prediccion_por_pais_cresta_bayesiana_mortalidad(pais):
    X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(days_since_1_22[ 150:], total_deaths[ 150:], test_size=0.05, shuffle=False)
    svm_death = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
    svm_death.fit(X_train_death, y_train_death)
    svm_pred_death = svm_death.predict(future_forcast)
    svm_test_pred_death = svm_death.predict(X_test_death)
    plt.plot(y_test_death)
    plt.plot(svm_test_pred_death)
    plt.legend(['Test Data', 'SVM Predictions'])
    print('MAE:', mean_absolute_error(svm_test_pred_death, y_test_death))
    print('MSE:',mean_squared_error(svm_test_pred_death, y_test_death))

    poly = PolynomialFeatures(degree=5)
    poly_X_train_death = poly.fit_transform(X_train_death)
    poly_X_test_death = poly.fit_transform(X_test_death)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_death = bayesian_poly.fit_transform(X_train_death)
    bayesian_poly_X_test_death = bayesian_poly.fit_transform(X_test_death)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_death, y_train_death)
    test_linear_pred = linear_model.predict(poly_X_test_death)
    linear_pred_death = linear_model.predict(poly_future_forcast)
    print('MAE:', mean_absolute_error(test_linear_pred, y_test_death))
    print('MSE:',mean_squared_error(test_linear_pred, y_test_death))

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
    bayesian_search.fit(bayesian_poly_X_train_death, y_train_death)
    bayesian_search.best_params_
    bayesian_death = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_death.predict(bayesian_poly_X_test_death)
    bayesian_pred = bayesian_death.predict(bayesian_poly_future_forcast)
    print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_death))
    print('MSE:',mean_squared_error(test_bayesian_pred, y_test_death))

    #plot_predictions(adjusted_dates, total_deaths, svm_pred_death, 'SVM Predictions', 'purple')

    plot_predictions(adjusted_dates, total_deaths, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana para ' + pais + ' sobre la mortalidad','purple')
    nombre_archivo = " predicciones_covid_" + 'Predicciones de Regresion Polinomeal para ' + pais + ' sobre la mortalidad' + '.png'
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid8 = base64.b64encode(image_file.read())
        print(encoded_string_covid8)
    return encoded_string_covid8

prediccion_por_pais_mortalidad_polinomeal('Guatemala')
prediccion_por_pais_cresta_bayesiana_mortalidad('Guatemala')

def prediccion_por_pais_cresta_bayesiana_recuperados(pais):
    X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(days_since_1_22[ 150:], total_recovered[ 150:], test_size=0.05, shuffle=False)
    poly = PolynomialFeatures(degree=5)
    poly_X_train_recovered = poly.fit_transform(X_train_recovered)
    poly_X_test_recovered = poly.fit_transform(X_test_recovered)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=5)
    bayesian_poly_X_train_recovered = bayesian_poly.fit_transform(X_train_recovered)
    bayesian_poly_X_test_recovered = bayesian_poly.fit_transform(X_test_recovered)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_recovered, y_train_recovered)
    test_linear_pred = linear_model.predict(poly_X_test_recovered)
    linear_pred_recovered = linear_model.predict(poly_future_forcast)
    print('MAE:', mean_absolute_error(test_linear_pred, y_test_recovered))
    print('MSE:',mean_squared_error(test_linear_pred, y_test_recovered))

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
    bayesian_search.fit(bayesian_poly_X_train_recovered, y_train_recovered)
    bayesian_search.best_params_
    bayesian_recovered = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_recovered.predict(bayesian_poly_X_test_recovered)
    bayesian_pred = bayesian_recovered.predict(bayesian_poly_future_forcast)
    print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_recovered))
    print('MSE:',mean_squared_error(test_bayesian_pred, y_test_recovered))
    #plot_predictions(adjusted_dates, total_recovered, linear_pred_death, 'Predicciones de Regresion Polinomeal para ' + pais + ' para Recuperados', 'purple')
    plot_predictions(adjusted_dates, total_recovered, bayesian_pred , 'Predicciones de regresión de la cresta bayesiana para ' + pais + ' sobre la recuperacion dentro del pais', 'purple')
    nombre_archivo = " predicciones_covid_" + 'Predicciones de regresión de la cresta bayesiana para ' + pais + ' sobre la recuperacion dentro del pais'+ '.png'
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid8 = base64.b64encode(image_file.read())
        print(encoded_string_covid8)
    return encoded_string_covid8

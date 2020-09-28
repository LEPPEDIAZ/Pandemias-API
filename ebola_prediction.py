import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import base64

# Sklearn
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
ebola= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Ebola/ebola_2014_2016_clean2.csv')
ebola

import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Ebola/ebola_2014_2016_clean2.csv', parse_dates=['Date'], index_col='Date')
df.rename(columns={'Cumulative no. of confirmed, probable and suspected cases': 'confirmed', 'Cumulative no. of confirmed, probable and suspected deaths': 'death'}, inplace=True)

# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.savefig("ebola_"+ title + ".png")
    

confirmado_por_pais =  df['Country']=='Guinea'
confirmado_por_pais = df[confirmado_por_pais]
confirmado_por_pais

plot_df(df, x=confirmado_por_pais.index, y=confirmado_por_pais.confirmed, title='Serie de Tiempo General para confirmados de Ebola') 
nombre_archivo = "ebola_"+ 'Serie de Tiempo General para confirmados de Ebola' + ".png"
with open(nombre_archivo, "rb") as image_file:
    encoded_ebola_pred = base64.b64encode(image_file.read())
    print(encoded_ebola_pred)
plot_df(df, x=confirmado_por_pais.index, y=confirmado_por_pais.death, title='Serie de Tiempo General para mortalidad de Ebola')
nombre_archivo = "ebola_"+ 'Serie de Tiempo General para mortalidad de Ebola' + ".png"
with open(nombre_archivo, "rb") as image_file:
    encoded_ebola_pred1 = base64.b64encode(image_file.read())
    print(encoded_ebola_pred1)  

df.reset_index(inplace=True)

# Prepare data
df['year'] = [d.year for d in df.Date]
df['month'] = [d.strftime('%b') for d in df.Date]
years = df['year'].unique()

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='confirmed', data=df, ax=axes[0])
sns.boxplot(x='month', y='confirmed', data=df.loc[~df.year.isin([1991, 2008]), :])

# Set Title
axes[0].set_title('Diagrama de Caja y Bigotes por año', fontsize=18); 
axes[1].set_title('Diagrama de Caja y Bigotes por mes', fontsize=18)
plt.savefig("ebola_"+ "diagramadecajaybigotes" + ".png")
nombre_archivo = "ebola_"+ "diagramadecajaybigotes" + ".png"
with open(nombre_archivo, "rb") as image_file:
    encoded_ebola_pred2 = base64.b64encode(image_file.read())
    print(encoded_ebola_pred2)

from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.confirmed.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

ads = df
ads = ads.dropna()
plt.figure(figsize=(15, 7))
plt.plot(ads.confirmed)
plt.title('Ads watched (hourly data)')
plt.grid(True)
#plt.show()
series = df.dropna()
series

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])

moving_average(ads.confirmed, 24) # prediction for the last observed day (past 24 hours)

def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
plotMovingAverage(ads.confirmed, 4) 

plotMovingAverage(ads.confirmed, 4, plot_intervals=True)

ads_anomaly = ads.confirmed.copy()
ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20] * 0.2 # say we have 80% drop of ads 

def weighted_average(series, weights):
    """
        Calculate weighter average on series
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1] * weights[n]
    return float(result)

weighted_average(ads.confirmed, [0.6, 0.3, 0.1])
# Creating a copy of the initial datagrame to make various transformations 
data = pd.DataFrame(ads.confirmed.copy())
data.columns = ["y"]
for i in range(6, 25):
    data["Date_{}".format(i)] = data.y.shift(i)

def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test
y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)



def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');

plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr());


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

dataset = df
dataset

dataset= dataset.dropna()

world_cases = np.array(dataset.confirmed).reshape(-1, 1)
total_deaths = np.array(dataset.death).reshape(-1, 1)
total_days = np.array(dataset.days_dif).reshape(-1, 1)
#world_cases =world_cases.dropna()
#total_deaths = total_deaths.dropna()
model = GaussianNB()
model.fit(world_cases, total_deaths )

expected = total_deaths
predicted = model.predict(world_cases)


print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Ebola/ebola_2014_2016_clean2.csv', parse_dates=['Date'], index_col='Date')
df

df = df.dropna()
df.rename(columns={'Cumulative no. of confirmed, probable and suspected cases': 'confirmed', 'Cumulative no. of confirmed, probable and suspected deaths': 'death'}, inplace=True)
x = df['death']
y = df['confirmed']
z = df['days_dif']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
model = GaussianNB()
x_train= np.array(x_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)
days_dif = np.array(z).reshape(-1, 1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred
accuracy = accuracy_score(y_test, y_pred)*100
accuracy
plt.figure(figsize=(16, 10))
plt.plot(df['days_dif'],  df['confirmed'])
plt.plot(y_pred)
plt.title('Predicciones con Bayes')
plt.title('Ebola', size=30)
plt.xlabel('Iniciando desde el dia 8/29/2014', size=30)
plt.ylabel('Cantidad de Casos', size=30)
plt.legend(['Casos Confirmados', 'Prediccion con Bayes'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.grid(True)
plt.savefig("ebola_"+ "prediccion_de_bayes" + ".png")
nombre_archivo = "ebola_"+ "prediccion_de_bayes" + ".png"
with open(nombre_archivo, "rb") as image_file:
    encoded_ebola_pred3 = base64.b64encode(image_file.read())
    print(encoded_ebola_pred3)

future_forcast = np.array([i for i in range(len(x)+50)]).reshape(-1, 1)
def plot_predictions(x, y, pred, algo_name, color):
    plt.figure(figsize=(16, 10))
    plt.plot(x, y)
    plt.plot(future_forcast, pred, linestyle='dashed', color=color)
    plt.title('COVID-19', size=30)
    plt.xlabel('Iniciando desde el dia 8/29/2014', size=30)
    plt.ylabel('Cantidad de Casos', size=30)
    plt.legend(['Casos', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    nombre_archivo = " predicciones_ebola_" + algo_name  + '.png'
    plt.savefig(nombre_archivo)
    ##plt.show()
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(total_days[50:], world_cases[50:], test_size=0.05, shuffle=False)
from sklearn.svm import SVR
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(y_test_confirmed)
plt.plot(svm_test_pred)
plt.legend(['Test Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge
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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
#plot_predictions(total_days, world_cases, svm_pred, 'Predicciones de SVM', 'purple')
plot_predictions(total_days, world_cases, linear_pred, 'Predicciones de Regresion Polinomeal Para Confirmados de Ebola', 'orange')
plot_predictions(total_days, world_cases, bayesian_pred, 'Predicciones de regresión de la cresta bayesiana Para Confirmados de Ebola', 'green')
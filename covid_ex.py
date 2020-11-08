# Data Management
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

confirmedcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
confirmedcovid.head()
ultimo_confirmado = confirmedcovid[confirmedcovid.columns[-1]]
print(len(ultimo_confirmado))
pais_confirmado = confirmedcovid[confirmedcovid.columns[1]]
print(len(ultimo_confirmado))
width = 0.35 
#confirmedcovid.plot.bar(x = 'Country/Region', y = '5/10/20')
order_confirmed = confirmedcovid.sort_values(confirmedcovid.columns[-1], ascending=True)
ultimo_confirmado_ordenado = order_confirmed[order_confirmed.columns[-1]]
#print("confirmados",ultimo_confirmado_ordenado)
ultimo_confirmado_ordenado_2 = order_confirmed[order_confirmed.columns[1]]
#print("paises", ultimo_confirmado_ordenado_2 )
confirmedcovid1 = confirmedcovid.dropna()
top_covid_confirmed = confirmedcovid1.nlargest(20, confirmedcovid1.columns[-1])
top_covid_confirmed.head()
top_covid_confirmed_graph = top_covid_confirmed.plot.barh(x = 'Province/State', y = top_covid_confirmed.columns[-1])
top_covid_confirmed_graph.figure.savefig('provinciasmasafectadasporcovid.png', bbox_inches='tight')
nombre_archivo = 'provinciasmasafectadasporcovid.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_ex_covid1 = base64.b64encode(image_file.read())
    print(encoded_string_ex_covid1)

print("Provincias Mas Afectadas")

confirmedcovid1 = confirmedcovid.dropna()
top_covid_confirmed = confirmedcovid1.nsmallest(20, confirmedcovid1.columns[-1])
top_covid_confirmed.head()
top_covid_confirmed_graph = top_covid_confirmed.plot.barh(x = 'Province/State', y = top_covid_confirmed.columns[-1])
print("Provincias Menos Afectadas")
top_covid_confirmed_graph.figure.savefig('provinciasmenosafectadasporcovid.png', bbox_inches='tight')
nombre_archivo = 'provinciasmenosafectadasporcovid.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_ex_covid2 = base64.b64encode(image_file.read())
    print(encoded_string_ex_covid2)

print("Provincias Mas Afectadas")


print("\n----------- Calcular media de confirmados -----------\n")
print(ultimo_confirmado.mean())
 
print("\n----------- Calcular mediana de confirmados -----------\n")
print(ultimo_confirmado.median())
 
print("\n----------- Calcular moda de confirmados -----------\n")
print(ultimo_confirmado.mode())

print("\n----------- Calcular moda de nombre confirmados -----------\n")
print(pais_confirmado.mode())

ax = ultimo_confirmado.plot.box()
print("MOST AFFECTED COUNTRY")
confirmedcovid.loc[confirmedcovid[confirmedcovid.columns[-1]].idxmax()]


print("Paises Menos Afectados")
ultimo_confirmado = confirmedcovid[confirmedcovid.columns[-1]]
clean_countries = confirmedcovid.groupby('Country/Region', as_index=False)[confirmedcovid.columns[-1]].sum()
clean_countries.head()
top_covid_confirmed = clean_countries.nsmallest(20, clean_countries.columns[-1])
top_covid_confirmed.head()
top_covid_confirmed_graph = top_covid_confirmed.plot.barh(x = 'Country/Region', y = top_covid_confirmed.columns[-1])
top_covid_confirmed_graph.figure.savefig('paisesmenosafectados.png', bbox_inches='tight')
nombre_archivo = 'paisesmenosafectados.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_ex_covid3 = base64.b64encode(image_file.read())
    print(encoded_string_ex_covid3)

print("Paises Mas Afectados")
ultimo_confirmado = confirmedcovid[confirmedcovid.columns[-1]]
clean_countries = confirmedcovid.groupby('Country/Region', as_index=False)[confirmedcovid.columns[-1]].sum()
clean_countries.head()
top_covid_confirmed = clean_countries.nlargest(20, clean_countries.columns[-1])
top_covid_confirmed.head()
top_covid_confirmed_graph = top_covid_confirmed.plot.barh(x = 'Country/Region', y = top_covid_confirmed.columns[-1])
top_covid_confirmed_graph.figure.savefig('paisesmasafectados.png', bbox_inches='tight')
nombre_archivo = 'paisesmasafectados.png'
with open(nombre_archivo, "rb") as image_file:
    encoded_string_ex_covid4 = base64.b64encode(image_file.read())
    print(encoded_string_ex_covid4)

totaldecontagios= ultimo_confirmado.sum()
totaldecontagios = int(totaldecontagios)
totaldecontagios = "{:,}".format(totaldecontagios)
print("TOTAL", totaldecontagios)

def totaldecontagios_def():
    confirmedcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    ultimo_confirmado = confirmedcovid[confirmedcovid.columns[-1]]
    totaldecontagios= ultimo_confirmado.sum()
    totaldecontagios = int(totaldecontagios)
    totaldecontagios = "{:,}".format(totaldecontagios)
    return totaldecontagios



def nombre_fecha_actual():
    global_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    columns = list(global_confirmed.columns)
    return columns[-1]
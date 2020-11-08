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

def proporcion_de_contagios():
    import base64
    plt.style.use(['dark_background'])
    plt.rcParams['axes.facecolor'] = 'black'
    #plt.figure(figsize=(20, 20))
    confirmedcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    ultimo_confirmado = confirmedcovid[confirmedcovid.columns[-1]]
    totaldecontagios= ultimo_confirmado.sum()
    totaldecontagios = int(totaldecontagios)
    #totaldecontagios = "{:,}".format(totaldecontagios)
    
    deathcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    ultimo_death = deathcovid[deathcovid.columns[-1]]
    totaldecontagiosdeath = ultimo_death.sum()
    totaldecontagiosdeath = int(totaldecontagiosdeath)
    #totaldecontagiosdeath = "{:,}".format(totaldecontagiosdeath)
    
    recoveredcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    ultimo_recovered = recoveredcovid[recoveredcovid.columns[-1]]
    totaldecontagiosrecovered= ultimo_recovered.sum()
    totaldecontagiosrecovered = int(totaldecontagiosrecovered)
    #totaldecontagiosrecovered = "{:,}".format(totaldecontagiosrecovered)

    labels = 'Casos sobre Confirmados: '+ str("{:,}".format(totaldecontagios)), 'Casos sobre Mortalidad: '+ str("{:,}".format(totaldecontagiosdeath)), 'Casos sobre recuperados: '+ str("{:,}".format(totaldecontagiosrecovered))
    sizes = [totaldecontagios, totaldecontagiosdeath, totaldecontagiosrecovered]
    explode = (0, 0, 0)  
    
    colors = ["green", "red", "blue"]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.title('COVID-19: Proporción entre recuperados , confirmados y casos de mortalidad' )
    nombre_archivo = "piechart covid.png"
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 8)
    plt.savefig(nombre_archivo)
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid7 = base64.b64encode(image_file.read())
        print(encoded_string_covid7)
    return encoded_string_covid7

def crecimiento_diario():
    import base64
    plt.style.use(['dark_background'])
    plt.rcParams['axes.facecolor'] = 'black'

    #plt.figure(figsize=(20, 20))
    confirmedcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    ultimo_confirmado = confirmedcovid[confirmedcovid.columns[-1]]
    ultimo_confirmado2 = confirmedcovid[confirmedcovid.columns[-2]]
    totaldecontagios= ultimo_confirmado.sum() - ultimo_confirmado2.sum()
    totaldecontagios = int(totaldecontagios)
    
    #print("ultimo confirmado" + str(ultimo_confirmado.sum()) + "ayer confirmado" + str(ultimo_confirmado2.sum()))
    #totaldecontagios = "{:,}".format(totaldecontagios)
    
    deathcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    ultimo_death = deathcovid[deathcovid.columns[-1]]
    ultimo_death2 = deathcovid[deathcovid.columns[-2]]
    totaldecontagiosdeath = ultimo_death.sum() - ultimo_death2.sum()
    totaldecontagiosdeath = int(totaldecontagiosdeath)
    #totaldecontagiosdeath = "{:,}".format(totaldecontagiosdeath)
    
    recoveredcovid= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    ultimo_recovered = recoveredcovid[recoveredcovid.columns[-1]]
    ultimo_recovered2 = recoveredcovid[recoveredcovid.columns[-1]]
    totaldecontagiosrecovered= ultimo_recovered.sum() - ultimo_recovered2.sum()
    totaldecontagiosrecovered = int(totaldecontagiosrecovered)
    #totaldecontagiosrecovered = "{:,}".format(totaldecontagiosrecovered)
    
    #print("ultimo confirmado" + str(ultimo_recovered.sum()) + "ayer confirmado" + str(ultimo_recovered.sum()))

    labels = 'Casos sobre Confirmados ', 'Casos sobre Mortalidad ', 'Casos sobre recuperados '
    sizes = [totaldecontagios, totaldecontagiosdeath, totaldecontagiosrecovered]
    explode = (0, 0, 0)  
    
    colors = ["green", "red", "blue"]
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    ax.bar(labels,sizes ,  color = colors)
      
    #plt.legend()
    columns = list(confirmedcovid.columns)
    plt.title('COVID-19: Aumento respecto al dia anterior: ' + columns[-2])
    nombre_archivo = "barplot covid.png"
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 8)
    plt.xlabel('Tipo de Caso')
    plt.ylabel('Cantidad')
    plt.savefig(nombre_archivo, bbox_inches='tight')
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid7 = base64.b64encode(image_file.read())
        #print(encoded_string_covid7)
    return encoded_string_covid7
    
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
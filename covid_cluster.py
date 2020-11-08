import numpy as np 
import matplotlib 
matplotlib.use('Agg')
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


def cluster_covid():
    df_in = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    df_in.iloc[:, 1:10].head()
    dates_vec = list(df_in.columns)[3:]
    average_time_vec = [None] * df_in.shape[0]

    for i, row_index in enumerate(df_in.index):

        weighted_sum, total_deaths = 0, 0
        
        for j, date in enumerate(dates_vec):
            current_term = df_in.at[row_index, date]
            weighted_sum += j * current_term
            total_deaths += current_term
        
        average_time_vec[i] = weighted_sum / total_deaths
        
    df_in['avg_time'] = average_time_vec

    n_lines = int((df_in.shape[0] * (df_in.shape[0] - 1)) / 2)
    list_country1, list_country2, list_w, list_d =\
        [None] * n_lines, [None] * n_lines, [None] * n_lines, [None] * n_lines

    line_index = 0
    epsilon = 0.001
    for i in range(0, df_in.shape[0] - 1):
        for j in range(i + 1, df_in.shape[0]):
            index_i, index_j = df_in.index[i], df_in.index[j]
            list_country1[line_index] = df_in.at[index_i, 'Country/Region']
            list_country2[line_index] = df_in.at[index_j, 'Country/Region']
            diff_time = df_in.at[index_i, 'avg_time'] - df_in.at[index_j,'avg_time']
            list_w[line_index] = (1 / (abs(diff_time) + epsilon))
            list_d[line_index] = abs(diff_time)
            line_index += 1
            
    df_graph = pd.DataFrame(dict(
        Country1 = list_country1,
        Country2 = list_country2,
        Weight = list_w,
        Distance = list_d
    ))

    df_graph
    graph_distance = nx.from_pandas_edgelist(df_graph, 'Country1', 'Country2', 'Distance')
    adj_matrix = nx.adjacency_matrix(graph_distance, weight='Distance')
    adj_matrix

    from sklearn.cluster import KMeans
    from matplotlib import pyplot as plt

    X = adj_matrix.toarray()
    distorsions = []
    #for k in range(2, 20):
    #    kmeans = KMeans(n_clusters=k)
    #    kmeans.fit(X)
    #    distorsions.append(kmeans.inertia_)

    #fig = plt.figure(figsize=(15, 5))
    #plt.plot(range(2, 20), distorsions)
    #plt.grid(True)
    #plt.title('The Elbow curve')

    from mpl_toolkits.mplot3d import Axes3D
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(X)
    kmeans.labels_
    y_kmeans = kmeans.predict(X)
    y_kmeans
    plt.style.use(['dark_background'])
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['text.color'] = 'white'
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Primer Nivel')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Segundo Nivel')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Tercer Nivel')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'orange', label = 'Cuarto Nivel')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'purple', label = 'Quinto Nivel')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'pink', label = 'Sexto Nivel')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title('Cluster COVID-19: Grupos de Transmision')
    plt.xlabel('Distancia')
    plt.ylabel('Cantidad')
    plt.legend()
    nombre_archivo = "cluster_covid.png"
    plt.savefig(nombre_archivo)
    with open(nombre_archivo, "rb") as image_file:
        encoded_string_covid7 = base64.b64encode(image_file.read())
        print(encoded_string_covid7)
    return encoded_string_covid7

cluster_covid()
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
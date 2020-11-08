from flask import Flask
#from vih_results import *
#from covid_results import *
from covid_ex import *
#from covid_cluster import *
from covid_general_predictions import *

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def hello_world():
    return 'Bienvenidos! este un API para analisis y predicciones de diferentes enfermedades con resultados de graficas en base 64'

#@app.route('/vih/comparacion_sexo')
#def vih_comparacion_sexo():
#    return encoded_string

#@app.route('/vih/comparacion_paternidad')
#def vih_comparacion_paternidad():
#    return encoded_string4

#@app.route('/vih/comparacion_hetero')
#def vih_comparacion_hetero():
#    return encoded_string3

#@app.route('/vih/comparacion_homo')
#def vih_comparacion_homo():
#    return encoded_string2

#@app.route('/vih/comparacion_contagios')
#def vih_compracion_contagios():
#    return encoded_string1

@app.route('/covid/confirmados_red_neuronal')
@cross_origin()
def covid_confirmados_red_neuronal():
    return predicciones_generales("RedesNeuronales")

@app.route('/covid/confirmados_red_bayesiana')
@cross_origin()
def covid_confirmados_red_bayesiana():
    return predicciones_generales("RegresionCrestaBayesiana")

@app.route('/covid/confirmados_lasso')
@cross_origin()
def covid_confirmados_lasso():
    return predicciones_generales("Lasso")

@app.route('/covid/mortalidad_red_bayesiana')
@cross_origin()
def covid_mortalidad_red_bayesiana():
    return predicciones_mortalidad_generales("RegresionCrestaBayesiana")

@app.route('/covid/mortalidad_red_neuronal')
@cross_origin()
def covid_mortalidad_red_neuronal():
    return predicciones_mortalidad_generales("RedesNeuronales")

@app.route('/covid/mortalidad_lasso')
@cross_origin()
def covid_mortalidad_lasso():
    return predicciones_mortalidad_generales("Lasso")


@app.route('/covid/recuperacion_red_bayesiana')
@cross_origin()
def covid_recuperacion_red_bayesiana():
    return predicciones_recuperacion_generales("RegresionCrestaBayesiana")

@app.route('/covid/recuperacion_red_neuronal')
@cross_origin()
def covid_recuperacion_red_neuronal():
    return predicciones_recuperacion_generales("RedesNeuronales")

@app.route('/covid/recuperacion_lasso')
@cross_origin()
def covid_recuperacion_lasso():
    return predicciones_recuperacion_generales("RedesNeuronales")


@app.route('/covid/confirmados_red_bayesiana_pais/<pais>')
@cross_origin()
def confirmados_red_bayesiana_pais(pais):
    grafica_prediccion = predicciones_por_pais("RegresionCrestaBayesiana",pais)
    #print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/confirmados_red_neuronal_pais/<pais>')
@cross_origin()
def confirmados_red_neuronal_pais(pais):
    grafica_prediccion = predicciones_por_pais("RedesNeuronales",pais)
    #print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/confirmados_lasso_pais/<pais>')
@cross_origin()
def confirmados_lasso_pais(pais):
    grafica_prediccion = predicciones_por_pais("Lasso",pais)
    #print(grafica_prediccion)
    return grafica_prediccion


@app.route('/covid/mortalidad_red_bayesiana_pais/<pais>')
@cross_origin()
def mortalidad_red_bayesiana_pais(pais):
    grafica_prediccion = predicciones_por_pais_mortalidad("RegresionCrestaBayesiana",pais)
    return grafica_prediccion


@app.route('/covid/mortalidad_red_neuronal_pais/<pais>')
@cross_origin()
def mortalidad_red_neuronal_pais(pais):
    grafica_prediccion = predicciones_por_pais_mortalidad("RedesNeuronales",pais)
    return grafica_prediccion


@app.route('/covid/mortalidad_lasso_pais/<pais>')
@cross_origin()
def mortalidad_lasso_pais(pais):
    grafica_prediccion = predicciones_por_pais_mortalidad("Lasso",pais)
    return grafica_prediccion


@app.route('/covid/recuperacion_red_bayesiana_pais/<pais>')
@cross_origin()
def recuperacion_red_bayesiana_pais(pais):
    grafica_prediccion = predicciones_por_pais_recuperacion("RegresionCrestaBayesiana",pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/recuperacion_red_neuronal_pais/<pais>')
@cross_origin()
def recuperacion_red_neuronal_pais(pais):
    grafica_prediccion = predicciones_por_pais_recuperacion("RedesNeuronales",pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/recuperacion_lasso_pais/<pais>')
@cross_origin()
def recuperacion_lasso_pais(pais):
    grafica_prediccion = predicciones_por_pais_recuperacion("Lasso",pais)
    print(grafica_prediccion)
    return grafica_prediccion


#@app.route('/covid/get_cluster')
#@cross_origin()
#def get_cluster():
#    print(cluster_covid())
#    return cluster_covid()


@app.route('/covid/total')
@cross_origin()
def covid_total_contagios():
    return totaldecontagios_def()

@app.route('/covid/fecha_actual')
@cross_origin()
def covid_fecha_actual():
    return nombre_fecha_actual()

@app.route('/covid/provinciasmasafectadas')
@cross_origin()
def covid_provincias_mas_afectadas():
    return encoded_string_ex_covid1

@app.route('/covid/provinciasmenosafectadas')
@cross_origin()
def covid_provincias_menos_afectadas():
    return encoded_string_ex_covid2

@app.route('/covid/paisesmasafectados')
@cross_origin()
def covid_paises_mas_afectados():
    return encoded_string_ex_covid4

@app.route('/covid/paisesmenosafectados')
@cross_origin()
def covid_paises_menos_afectados():
    return encoded_string_ex_covid3




if __name__ == '__main__':
    app.run()
from flask import Flask
from vih_results import *
from covid_results import *
from covid_ex import *
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def hello_world():
    return 'Bienvenidos! este un API para analisis y predicciones de diferentes enfermedades con resultados de graficas en base 64'

@app.route('/vih/comparacion_sexo')
def vih_comparacion_sexo():
    return encoded_string

@app.route('/vih/comparacion_paternidad')
def vih_comparacion_paternidad():
    return encoded_string4

@app.route('/vih/comparacion_hetero')
def vih_comparacion_hetero():
    return encoded_string3

@app.route('/vih/comparacion_homo')
def vih_comparacion_homo():
    return encoded_string2

@app.route('/vih/comparacion_contagios')
def vih_compracion_contagios():
    return encoded_string1

@app.route('/covid/confirmados_polinomeal')
@cross_origin()
def covid_confirmados_polinomeal():
    return encoded_string_covid1

@app.route('/covid/confirmados_red_bayesiana')
@cross_origin()
def covid_confirmados_red_bayesiana():
    return encoded_string_covid2

@app.route('/covid/mortalidad_polinomeal')
@cross_origin()
def covid_mortalidad_polinomeal():
    return encoded_string_covid3

@app.route('/covid/mortalidad_red_bayesiana')
@cross_origin()
def covid_mortalidad_red_bayesiana():
    return encoded_string_covid4

@app.route('/covid/recuperacion_polinomeal')
@cross_origin()
def covid_recuperacion_polinomeal():
    return encoded_string_covid5

@app.route('/covid/recuperacion_red_bayesiana')
@cross_origin()
def covid_recuperacion_red_bayesiana():
    return encoded_string_covid6

@app.route('/covid/confirmados_polinomeal_pais/<pais>')
@cross_origin()
def confirmados_polinomeal_pais(pais):
    grafica_prediccion = prediccion_por_pais_confirmados_polinomeal(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/confirmados_red_bayesiana_pais/<pais>')
@cross_origin()
def confirmados_red_bayesiana_pais(pais):
    grafica_prediccion = prediccion_por_pais_cresta_bayesiana(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/mortalidad_polinomeal_pais/<pais>')
@cross_origin()
def mortalidad_polinomeal_pais(pais):
    grafica_prediccion = prediccion_por_pais_mortalidad_polinomeal(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/mortalidad_red_bayesiana_pais/<pais>')
@cross_origin()
def mortalidad_red_bayesiana_pais(pais):
    grafica_prediccion = prediccion_por_pais_cresta_bayesiana_mortalidad(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/recuperacion_red_bayesiana_pais/<pais>')
@cross_origin()
def recuperacion_red_bayesiana_pais(pais):
    grafica_prediccion = prediccion_por_pais_cresta_bayesiana_recuperados(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/total')
@cross_origin()
def covid_total_contagios():
    return totaldecontagios

@app.route('/covid/fecha_actual')
@cross_origin()
def covid_fecha_actual():
    return nombre_fecha_actual

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
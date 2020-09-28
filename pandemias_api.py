from flask import Flask
from vih_results import *  
from covid_results import *
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Bienvenidos este un API para analisis y predicciones de diferentes pandemias con resultados de graficas en base 64'

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
def covid_confirmados_polinomeal():
    return encoded_string_covid1

@app.route('/covid/confirmados_red_bayesiana')
def covid_confirmados_red_bayesiana():
    return encoded_string_covid2

@app.route('/covid/mortalidad_polinomeal')
def covid_mortalidad_polinomeal():
    return encoded_string_covid3

@app.route('/covid/mortalidad_red_bayesiana')
def covid_mortalidad_red_bayesiana():
    return encoded_string_covid4

@app.route('/covid/recuperacion_polinomeal')
def covid_recuperacion_polinomeal():
    return encoded_string_covid5

@app.route('/covid/recuperacion_red_bayesiana')
def covid_recuperacion_red_bayesiana():
    return encoded_string_covid6

@app.route('/covid/confirmados_polinomeal_pais/<pais>')
def confirmados_polinomeal_pais(pais):
    grafica_prediccion = prediccion_por_pais_confirmados_polinomeal(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/confirmados_red_bayesiana_pais/<pais>')
def confirmados_red_bayesiana_pais(pais):
    grafica_prediccion = prediccion_por_pais_cresta_bayesiana(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/mortalidad_polinomeal_pais/<pais>')
def mortalidad_polinomeal_pais(pais):
    grafica_prediccion = prediccion_por_pais_mortalidad_polinomeal(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/mortalidad_red_bayesiana_pais/<pais>')
def mortalidad_red_bayesiana_pais(pais):
    grafica_prediccion = prediccion_por_pais_cresta_bayesiana_mortalidad(pais)
    print(grafica_prediccion)
    return grafica_prediccion

@app.route('/covid/recuperacion_red_bayesiana_pais/<pais>')
def recuperacion_red_bayesiana_pais(pais):
    grafica_prediccion = prediccion_por_pais_cresta_bayesiana_recuperados(pais)
    print(grafica_prediccion)
    return grafica_prediccion




if __name__ == '__main__':
    app.run()
from flask import Flask
from vih_results import *  
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

if __name__ == '__main__':
    app.run()
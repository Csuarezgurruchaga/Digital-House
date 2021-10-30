import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import sys
sys.path.insert(1,  '../../../common')
from checkpoint import *

descripcion_1 = '''
Durante el entrenamiento de un modelo de clasificación:
'''
opciones_1 = ['El modelo aprende a predecir un valor numérico dado un conjunto de atributos',
              'El modelo aprende a predecir un valor categórico dado un conjunto de atributos',
              'El modelo busca agrupar los datos según su cercanía',
              'Ninguna de las anteriores es correcta']
feedback_1 = ['Si predijera un valor numérico, más que un modelo de clasificación, sería un modelo de regresión.',
              '¡Exacto!',
              'Aunque hay modelos de clasificación que se basan en nociones de cercanía o distancia, como KNN, el objetivo de un clasificador no es agrupar los datos más similares entre sí.',
              '¡Repasá bien las opciones porque hay una que es correcta!']
test_1 = create_multiple_choice(descripcion_1, opciones_1, opciones_1[1], feedback_1)


descripcion_2 = '''
KNN es un modelo basado en instancias porque:
'''
opciones_2 = ['Entrena con instancias, es decir, con ejemplos',
              'Generaliza lo que aprende de las instancias con las que entrena',
              'Memoriza las instancias de entrenamiento y luego clasifica datos nuevos en base a las instancias memorizadas',
              'Ninguna de las anteriores es correcta']
feedback_2 = ['¿Existe algún modelo de machine learning que no aprenda con ejemplos?',
              'Cualquier modelo predictivo se va a encargar de generalizar a partir de datos de entrenamiento',
              '¡Perfecto! Vimos que KNN compara cada dato de test contra todos los datos de train para poder detectar los vecinos más cercanos, de ahí la noción de modelo basado en instancias.',
              '¡Repasá bien las opciones porque hay una que es correcta!']
test_2 = create_multiple_choice(descripcion_2, opciones_2, opciones_2[2], feedback=feedback_2)


descripcion_3 = '''
El hiperparámetro k de KNN (en Scikit - Learn, `n_neighbors`):
'''
opciones_3 = ['Hace referencia a la cantidad de instancias que va a memorizar',
              'Aumenta el sesgo del modelo',
              'Hace referencia a la cantidad de vecinos que considera para clasificar cada nueva observación',
              'Define qué caso particular de métrica de distancia de Minkowsky se va a considerar']
feedback_3 = ['KNN memoriza todas las instancias del set de entrenamiento.',
              'Si bien típicamente cuanto mayor sea el k, mayor será el sesgo del modelo, esto no significa que el valor de k per se esté asociado a un mayor sesgo.',
              '¡Muy bien! k es el hiperparámetro del modelo que define cuántos vecinos se van a considerar al momento de hacer las predicciones.',
              'En este caso, no se trata de k sino de p.']

test_3 = create_multiple_choice(descripcion_3, opciones_3, opciones_3[2], feedback_3)


descripcion_4 = '''
Si cada "cuadra" de este callejero cuadriculado tiene una longitud de 1, ¿cuál es la distancia euclídea entre los puntos? ¿Y la distancia Manhattan?
'''
opciones_4 = ['Distancia euclídea = 12, distancia Manhattan = 8.49',
              'Distancia euclídea = 36, distancia Manhattan = 12',
              'Distancia euclídea = 8.49, distancia Manhattan = 12',
              'Distancia euclídea = 12, distancia Manhattan = 12']
feedback_4 = ['¡Revisá las fórmulas!',
              '¡Revisá las fórmulas!',
              '¡Correcto!',
              '¡Revisá las fórmulas!']

def test_4():
    response = requests.get('https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Manhattan_distance.svg/800px-Manhattan_distance.svg.png')
    img = Image.open(BytesIO(response.content))
    basewidth = 350
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    display(img)
    create_multiple_choice(descripcion_4, opciones_4, opciones_4[2], feedback_4)()


descripcion_5 = """
Considere la siguiente matriz de un confusión. ¿Cuál es el accuracy del modelo?
"""
opciones_5 = ['70/120', '60/90', '10/30', '50/120']
feedback_5 = ['¡Correcto!',
              'Recordá que el accuracy mide la proporción de clasificaciones corectas sobre el total.',
              'Recordá que el accuracy mide la proporción de clasificaciones corectas sobre el total.',
              'Recordá que el accuracy mide la proporción de clasificaciones corectas sobre el total.']

def test_5():

    fig = plt.figure()
    
    tabla = plt.table(cellText = np.array([[60, 20, 80],
                                           [30, 10, 40],
                                           [90, 30, 120]]),
                      cellLoc = 'center',
                      colLabels=['Predice clase negativa', 'Predice clase positiva', 'Total'],
                      rowLabels=['Clase negativa', 'Clase positiva', 'Total'],
                      loc='left')
    tabla.scale(2, 5)
    tabla.set_fontsize(30)
    plt.axis('off');
    
    create_multiple_choice(descripcion_5, opciones_5, opciones_5[0], feedback_5, fig=fig)()
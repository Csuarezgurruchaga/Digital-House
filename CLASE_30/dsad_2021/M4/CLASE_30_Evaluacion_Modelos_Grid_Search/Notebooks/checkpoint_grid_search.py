import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import sys

sys.path.insert(1, '../../../common')
from checkpoint import *

descripcion_1 = '''
La estrategia de GridSearch:
'''
opciones_1 = ['Hace una búsqueda exhaustiva de los parámetros del modelo.',
              'Hace una búsqueda exhaustiva de los hiperparámetros del modelo.', 
              'Hace una búsqueda aleatoria de los hiperparámetros del modelo.']
feedback_1 = ['Recordá que los parámetros de un modelo son características del modelo cuyo valor se estima a partir del entrenamiento con los datos.',
              '¡Muy bien!.',
              'La estrategia que hace una búsqueda aleatoria es RandomizedSearch.']
test_1 = create_multiple_choice(descripcion_1, opciones_1, opciones_1[1], feedback_1)



descripcion_2 = '''
Estamos planteando un problema de machine-learning de qué tipo?:
'''
opciones_2 = ['Clasificación',
              'Regresión',
              'Reducción de la dimensionalidad']
feedback_2 = ['¡Muy bien! En los problemas de clasificación el objetivo es predecir la pertenencia o la probabilidad de pertenencia de un caso a una clase.',
              'Recordá que en la regresión el valor a predecir pertenece a una variable continua; sería el caso si en vez de niveles de puntaje, tuviéramos los puntajes brutos.',
              'Recordá que PCA es, por ejemplo, una técnica para reducir la dimensionalidad de un set de datos.']
test_2 = create_multiple_choice(descripcion_2, opciones_2, opciones_2[0], feedback=feedback_2)



descripcion_3 = '''
¿Cómo se realizará el proceso de búsqueda?:
'''
opciones_3 = ['En forma exhaustiva',
              'En forma aleatoria']
feedback_3 = ['¡Muy bien!',
              'Recordá que es RandomizedSearch la estrategia que nos permite recorrer el espacio de hiperparámetros en forma aleatoria.']
test_3 = create_multiple_choice(descripcion_3, opciones_3, opciones_3[0], feedback=feedback_3)




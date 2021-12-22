import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import sys

sys.path.insert(1, '../../../common/')
from checkpoint import *

descripcion_1 = '''
Para construir un Pipeline se deben pasar:
'''
opciones_1 = ['Un diccionario que contiene el nombe de cada paso y instancias de los estimadores.',
              'Una lista de tuplas que contienen el nombre de cada paso y una instancia de un estimador.', 
              'Una instancia de GridSearchCV.']
feedback_1 = ['Nop. Recordá que le tenemos que pasar una lista de tuplas que contienen el nombre de cada paso y una instancia de un estimador.',
              '¡Muy bien!',
              'Nop. Recordá que le tenemos que pasar una lista de tuplas que contienen el nombre de cada paso y una instancia de un estimador. Luego vamos a ver que los Pipelines los podemos combinar con GridSearchCV']
test_1 = create_multiple_choice(descripcion_1, opciones_1, opciones_1[1], feedback_1)



descripcion_2 = '''
Cuando aplicamos los métodos .score o .predict de un Pipeline sobre los datos de test:
'''
opciones_2 = ['Desde el paso inicial hasta el anteúltimo se aplica el método .transform de cada uno de los objetos transformadores.',
              'Desde el paso inicial hasta el anteúltimo se aplica el método .fit de cada uno de los objetos transformadores.',
              'Desde el paso inicial hasta el anteúltimo se aplica el método .fit_transform de cada uno de los objetos transformadores.']
feedback_2 = ['¡Muy bien!',
              'Recordá que el .fit y el .transform sólo se van a aplicar en conjunto a los datos de entrenamiento, mientras que a los de test sólo se les va a aplicar el .transform.',
              'Recordá que el .fit y el .transform sólo se van a aplicar en conjunto a los datos de entrenamiento, mientras que a los de test sólo se les va a aplicar el .transform.']
test_2 = create_multiple_choice(descripcion_2, opciones_2, opciones_2[0], feedback=feedback_2)



descripcion_3 = '''
Combinar Pipeline con GridSearchCV:
'''
opciones_3 = ['Nos permite evaluar cuáles son los mejores pasos de preprocesamiento combinados con la mejor configuración de hiperparámetros del modelo',
              'Únicamente nos permite identificar la mejor combinación de hiperparámetros del modelo']
feedback_3 = ['¡Muy bien!',
              'Nop. Recordá que nos permite evaluar cuáles son los mejores pasos de preprocesamiento combinados con la mejor configuración de hiperparámetros del modelo.']
test_3 = create_multiple_choice(descripcion_3, opciones_3, opciones_3[0], feedback=feedback_3)




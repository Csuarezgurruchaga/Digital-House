import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import sys

sys.path.insert(1, '../../../common')
from checkpoint import *

descripcion_1 = '''
Se tiene un conjunto de imágenes médicas en las que aparecen tumores y se quiere automatizar la clasificación de 
los mismos. Sabiendo que pueden ser malignos o benignos, ¿qué tipo de problema es?
'''
opciones_1 = ['Clasificación binaria', 'Clasificación multiclase', 'Clasificación multietiqueta']
feedback_1 = ['¡Muy bien! Al haber sólo dos clases posibles, es una clasificación binaria.',
              'Clasificación multiclase significa que hay más de dos clases posibles para asignar.',
              'Multietiqueta implica que una observación puede pertenecer a más de una clase a la vez.']
test_1 = create_multiple_choice(descripcion_1, opciones_1, 'Clasificación binaria', feedback_1)

descripcion_2 = '''
Un banco posee un detector de clientes fraudulentos que clasifica a las operaciones como "legales" (0) o 
"sospechosas" (1).
Últimamente, varias operaciones legales fueron clasificadas como sospechosas, haciendo que el banco pierda tiempo 
y dinero investigando operaciones legales.
Como responsable del área de Data Science, te exigen una solución rápida. ¿Cuál de estas alternativas probarías 
primero?
'''
opciones_2 = ['Recolectar datos nuevos y reentrenar el modelo',
              'Elevar el valor de umbral',
              'Reducir el valor de umbral',
              'Usar los mismos datos para entrenar otro modelo más complejo']
feedback_2 = ['Esto tomaría demasiado tiempo.',
              '¡Muy Bien! Utilizando el mismo modelo y elevando el umbral, estamos siendo más "exigentes" para \nclasificar una operación como fraudulenta',
              'Esta es una opción rápida que haría que más operaciones sean clasificadas como fraudulentas',
              'Podría funcionar, pero el entrenamiento de un nuevo modelo podría tomar mucho tiempo.']
test_2 = create_multiple_choice(descripcion_2, opciones_2, opciones_2[1], feedback=feedback_2)

descripcion_3 = '''
¿Cuál de estas funciones es la que se utiliza para estimar la probabilidad de que una observación pertenezca a una clase?
'''
opciones_3 = ['Sigmoide', 'ReLu', 'tanh', 'lineal']
feedback_3 = ['¡Muy bien! Esta función está acotada en (0,1) y es la que utiliza la regresión logística',
              'Si bien esta función no puede tomar valores negativos, pero puede ser mayor a 1.',
              'Si bien esta función parece similar a la correcta, está acotada en el rango (-1,1).',
              'Este sería el caso de una regresión lineal, que no sirve para problemas de clasificación.']


def test_3():
    x = np.linspace(-5, 5)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    linear = x
    relu = np.clip(x, a_min=0, a_max=None)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].plot(x, linear)
    ax[0, 0].set_title('Lineal')
    ax[0, 1].plot(x, tanh)
    ax[0, 1].set_title('tanh')
    ax[1, 0].plot(x, sigmoid)
    ax[1, 0].set_title('Sigmoide')
    ax[1, 1].plot(x, relu)
    ax[1, 1].set_title('ReLu')
    for a in ax.ravel():
        a.set_xlabel('x')
        a.set_ylabel('y')
        a.grid(True)
    create_multiple_choice(descripcion_3, opciones_3, 'Sigmoide', feedback_3, fig=fig)()


descripcion_4 = '''
Dado un problema de clasificación binaria donde las observaciones tienen 2 features x1 y x2, graficamos los datos
y obtenemos la siguiente distribución. ¿Podemos utilizar una regresión logística sobre este dataset?  
'''
opciones_4 = ['Sí.', 'No.']
feedback_4 = ['Recordá qué forma tiene el umbral de decisión cuando los datos tienen 2 features.',
              '¡Muy bien! Con 2 features sólo podemos formar una recta y no sirve para este caso.']


def test_4():
    x, y = make_circles(n_samples=400, factor=0.4, noise=0.1)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow')
    plt.xlabel('x1')
    plt.ylabel('x2')
    create_multiple_choice(descripcion_4, opciones_4, 'No.', feedback_4, fig=fig)()


descripcion_5 = """
En los problemas de clasificación, a diferencia de los problemas de regresión:  
"""
opciones_5 = [
    'No hay que preocuparse por el overfitting.',
    'No hay que preocuparse por el underfitting.',
    'La regularización pierde sentido.',
    'El MSE, RMSE, MAE y R2 pierden sentido.'
]

feedback_5 = [
    '''
    El overfitting es un caso al que hay que prestarle atención sin importar si la variable objetivo 
    es categórica o numérica
    ''',
    '''
    El underfitting es un caso al que hay que prestarle atención sin importar si la variable objetivo 
    es categórica o numérica
    ''',
    '''
    La regularización ayudará a que los coeficientes asociados a cada variable no dependan del rango de las 
    mismas y a que el algoritmo converja más rápidamente. En la implementación de Scikit-Learn se aplica por defecto.
    ''',
    '''
    ¡Muy bien! Estas métricas se utilizan cuando la variable objetivo es continua y podemos cuantificar el 
    error del modelo para cada observación. En los casos de clasificación, se utilizan otras métricas como 
    el accuracy para evaluar los modelos.
    '''
]
test_5 = create_multiple_choice(descripcion_5, opciones_5, opciones_5[-1], feedback_5)

descripcion_6 = '''
Se quiere resolver un problema de clasificación binaria, para lo cual se cuenta con un dataset con 1000 casos de 
entrenamiento (X_train) y 100 casos de test (X_test). Luego de entrenar el modelo con Scikit-Learn, qué output 
tendrá la siguiente línea de código?
lr.predict_proba(X_test). 
'''

opciones_6 = [
    '(2, 100)',
    '(100, 2)',
    '(100, 1)',
    '(1000,2)',
    'No se puede saber sin conocer la cantidad de features.'
]

feedback_6 = [
    '''
    Recordá que la forma de un array de 2 dimensiones es (filas, columnas) y solemos organizar el dataset
    en una observación por fila.
    ''',
    '''
    ¡Muy bien! Son 100 observaciones (filas) en el set de testeo y las columnas indican la probabilidad
     de pertenencia a cada clase (2 clases).
    ''',
    '''
    ¿Cuántas clases posibles hay en una clasificación binaria?
    ''',
    '''
    Recordá que solemos organizar el dataset en una observación por fila. ¿Cuántas observaciones hay en el
    set de testeo?
    ''',
    '''
    Recordá que el método .predict_proba(X) devuelve las probabilidades de pertenencia de cada observación
    a cada clase.
    '''
]

test_6 = create_multiple_choice(descripcion_6, opciones_6, opciones_6[1], feedback_6)

descripcion_7 = '''
Luego de ejecutar el siguiente código:

lr = LogisticRegression()
lr.fit(X,y)

Seleccioná cuáles de las siguientes afirmaciones son correctas, sabiendo que 'y' puede tomar 3 valores
posbiles:
'''

opciones_7 = [
    'lr.predict_proba(X).sum(axis=1) devuelve un array lleno de 1s',
    'lr.predict_proba(X) > 0.5 == lr.predict(X).astype(bool) evalúa siempre positivo',
    'lr.predict_proba(X).argmax(axis=1) == lr.predict(X) evalúa siempre positivo',
    'El código arrojará un error ya que la clase LogisticRegression() sólo admite problemas binarios. '
]

test_7 = create_checkbox(descripcion_7, opciones_7, opciones_7[::2])

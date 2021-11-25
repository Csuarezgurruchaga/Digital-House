import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import sys

sys.path.insert(1, '../../../common')
from checkpoint import *

descripcion_1 = '''
La matriz de confusión muestra:
'''
opciones_1 = ['Los resultados observados vs. los predichos por un modelo.',
              'Los resultados estimados por el modelo vs. los resultados predichos.', 
              'El accuracy del modelo.']
feedback_1 = ['¡Muy bien! La matriz de confusión es una tabla de doble entrada que nos permite comparar los casos observados con los predichos por un modelo.',
              'La matriz de confusión siempre contiene los casos observables o reales y los predichos o estimados por el modelo.',
              'El accuracy se calcula a partir de la matriz de confusión, pero no viene dado en ella.']
test_1 = create_multiple_choice(descripcion_1, opciones_1, opciones_1[0], feedback_1)



descripcion_2 = '''
¿Cuántos True Positive (TP) obtuvimos con nuestro modelo?
'''
opciones_2 = ['3464',
              '283',
              '305',
              '898']
feedback_2 = ['Esta es la cantidad de True Negative (TN).',
              '¡Muy Bien! En nuestro modelo identificamos correctamente 283 casos de la clase positiva.',
              'Esta es la cantidad de False Positive (FP).',
              'Esta es la cantidad de False Negative (FN).']
test_2 = create_multiple_choice(descripcion_2, opciones_2, opciones_2[1], feedback=feedback_2)



descripcion_3 = '''
TP/(TP+FP) es la fórmula de:
'''
opciones_3 = ['Specificity',
              'Recall',
              'Precision',
              'Sensitivity']
feedback_3 = ['La fórmula de Specificity es TN/(FP+TN), es decir cuántos casos negativos identificamos correctamente del total de casos negativos.',
              'La fórmula de Recall es TP/(TP+FN), es decir cuántos casos positivos identificamos del total de casos positivos (esta métrica también se llama Sensitivity.',
              '¡Muy Bien! Precision nos indica cuantos casos positivos identificamos correctamente del total de casos que identificamos como positivos.',
              'La fórmula de Sensitivity es TP/(TP+FN), es decir cuántos casos positivos identificamos del total de casos positivos (esta métrica también se llama Recall).']
test_3 = create_multiple_choice(descripcion_3, opciones_3, opciones_3[2], feedback=feedback_3)




descripcion_4 = '''
La CURVA ROC nos permite evaluar la relación entre:
'''
opciones_4 = ['Recall y 1-specificity',
              'Recall y sensitivity',
              'Recall y F1',
              'Recall y precision']
feedback_4 = ['La Curva ROC nos permite ver la relación entre los verdaderos positivos y falsos negativos (false positive rate o 1-specificity).',
              'Estos dos términos son sinónimos, y se refieren a qué tan bien nuestro modelo identifica casos positivos sobre el total de casos positivos.',
              'La Curva ROC sí incluye el Recall (verdaderos positivos) pero no el F1 (la media armónica entre precision y recall).',
              'La Curva ROC sí incluye el Recall (verdaderos positivos) pero no precision (cantidad de casos positivos identificados, del total de casos positivos predichos).']
test_4 = create_multiple_choice(descripcion_4, opciones_4, opciones_4[0], feedback=feedback_4)




import numpy
import math
import csv

import tensorflow as tf

from tf.keras.models import Sequential
from tf.keras.layers import Dense, Softmax


def sigmoid(inpt):
    return 1.0/(1.0+numpy.exp(-1*inpt))

def predict_outputs(weights_mat, data_inputs, data_outputs, activation="relu"):
    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights_mat:
            r1 = numpy.matmul(a=r1, b=curr_weights)
            if activation == "sigmoid":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    accuracy = (correct_predictions/data_outputs.size)*100
    return accuracy, predictions
    

def calculate_distance
    a= data_inputs.
    b=data_outputs.
dst = distance.euclidean(a, b)

def otroFitness(weights_mat, data_inputs, data_outputs, activation="sigmoid"):
    Error=dist(data_inputs,data_outputs)
#la función de fitness debe ser medir la distancia de la clasificación que hace la red con
#la clasificación que tienen los datos
#Dependiendo del rendimiento del fitness, seleccionas los mejores cromosomas para la reproducción del genético.

    model = Sequential()
    model.add(Dense(6, input_shape=(14,)))
    model.add(Dense(3, input_shape=(6,)))
    model.add(Softmax())


# *******************************************************************

def iterVinos():
    with open('mlptrain.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count in [0,1]:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                yield row
        print(f'Processed {line_count} lines.')


def fitness(individuo):
# El individuo son los pesos de la red acomodados como una lista de listas
# 
# Individuo = [ 
#   [3 umbrales], 
#   [ [13 pesos por cada feature]*6 una lista por cada neurona ], 
#   [ [3 pesos por cada neuronaOculta]*3 una lista por cada neurona final]  ]
# 
# Los datos son los vinos del archivo .csv
# 
# 

    # Error acumelado
    error = 0
    cuentaVinos = 0

    # Valores del individuo
    umbrales = individuo[0]
    oculto_ws = individuo[1]
    final_ws = individuo[2]



    # Iterador de datos
    itera = iterVinos()


# CALCULO de NEURONAS

    # iteramos sobre los datos
    for wine in itera:

        cuentaVinos += 1

        # Valores de las características de los vinos
        feats = np.asarray([ int(x) for x in wine[0:13] ])

        # Valores en Neuronas para Vino actual
        capaNeuronasOcultas = [0.0 for x in oculto_ws ]
        capaNeuronasFinal = [0.0 for x in final_w ]
        capaUmbralizada = [0.0 for x in umbrales]


        # Calculamos valores de capa oculta para el vino actual
        for i, pesosNeurona in enumerate(oculto_ws):
            
            # pesosNeurona es la lista de pesos correspondientes a esa neurona
            pesos = np.asarray(pesosNeurona)
            neurona = sum(feats*pesos)
            activada = sigmoid(neurona)
            capaNeuronasOcultas[i] = activada


        # Calculamos valores de última capa a partir de la oculta, para el vino actual
        for i, pesosNeurona in enumerate(final_ws):
            
            capaAnterior = np.asarray(capaNeuronasOcultas)
            
            pesos = np.asarray(pesosNeurona)
            neurona = sum(capaAnterior*pesos)
            activada = sigmoid(neurona)
            capaNeuronasFinal[i] = activada

        # Volvemos 0 las neuronas de la última capa que no hayan pasado el umbral
        for i, x in enumerate(capaNeuronasFinal):
            if x > umbrales[i]:
                capaUmbralizada[i] = x

        # Calculamos la predicción del individuo (redN) con ESTOS cromosomas para el vino actual
        prediction = np.argmax(capaUmbralizada)

        # Sumamos el error, si lo hay
        if not wine[prediction + 13] == 1:
            error += 1


    fitness = error/cuentaVinos

    return fitness








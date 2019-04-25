import numpy
import math

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

def fitness(weights_mat, data_inputs, data_outputs, activation="sigmoid"):
    Error=dist(data_inputs,data_outputs)
#la función de fitness debe ser medir la distancia de la clasificación que hace la red con
#la clasificación que tienen los datos
#Dependiendo del rendimiento del fitness, seleccionas los mejores cromosomas para la reproducción del genético.

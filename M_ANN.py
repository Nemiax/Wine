import numpy as np
import csv



def sigmoid(inpt):
    return 1.0/(1.0+numpy.exp(-1*inpt))

def vec2individuo( vec, nUmbrales, nFeatures, nOcultas, nSalida ):

    last = 0

    individuo = []

    # Umbrales.
    umbrales = vec[0:nUmbrales]
    last += nUmbrales
    individuo += [umbrales]


    # Pesos Capa Oculta
    pesosCapaOculta = []

    for i in range(nOcultas):
        pesos = vec[last:last+nFeatures]
        last += nFeatures
        pesosCapaOculta += [pesos]

    individuo += [pesosCapaOculta]


    # Pesos Capa Final (de Salida)
    pesosCapaFinal = []

    for i in range(nSalida):
        pesos = vec[last:last+nOcultas]
        last += nOcultas
        pesosCapaFinal += [pesos]

    individuo += [pesosCapaFinal]

    return individuo




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
#   [ [6 pesos por cada neuronaOculta]*3 una lista por cada neurona final]  ]
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








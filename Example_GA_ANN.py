import csv
filename = "mlptrain.csv"
#La base de datos está en csv, son 13 características de los vinos. Se supone tome el primer renglón como las entradas 
#(e_inputs) y multiplique pesos aleatorios y empiece a evolucionar.
#Problema 1: no econtré cómo hacer eso :(


"""
Genetic algorithm parameters:
    Mating Pool Size (Number of Parents)
    Population Size
    Number of Generations
    Mutation Percent
"""
e_inputs=13  
num_parents_mating = 4
num_generations = 100
mutation_percent = 10

#Creating the initial population.
initial_pop_weights = []
for curr_sol in numpy.arange(0, e_inputs):
    HL1_neurons = 13 #tengo 13 entradas
    input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                             size=(data_inputs.shape[1], HL1_neurons))
    
    
    output_neurons = 3 #tres salidas, pues hay tres clasificaciones de vinos
    HL1_output_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                              size=(HL1_neurons, output_neurons))

    initial_pop_weights.append(numpy.array([input_HL1_weights,
                                                HL1_output_weights]))




pop_weights_mat = numpy.array(initial_pop_weights)
pop_weights_vector = GA.mat_to_vector(pop_weights_mat)

best_outputs = []
accuracies = numpy.empty(shape=(num_generations))

for generation in range(num_generations):
    print("Generation : ", generation)

    # converting the solutions from being vectors to matrices.
    pop_weights_mat = GA.vector_to_mat(pop_weights_vector,
                                       pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = ANN.fitness(pop_weights_mat,
                          data_inputs,
                          data_outputs,
                          activation="sigmoid")
    accuracies[generation] = fitness[0]
    print("Fitness")
    print(fitness)

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(pop_weights_vector,
                                    fitness.copy(),
                                    num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents,
                                       offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover,
                                     mutation_percent=mutation_percent)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

pop_weights_mat = GA.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat [0, :]
acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")
print("Accuracy of the best solution is : ", acc)

matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=20)
matplotlib.pyplot.ylabel("Fitness", fontsize=20)
matplotlib.pyplot.xticks(numpy.arange(0, num_generations+1, 100), fontsize=15)
matplotlib.pyplot.yticks(numpy.arange(0, 101, 5), fontsize=15)



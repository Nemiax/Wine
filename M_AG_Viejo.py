import numpy
import M_GA
import M_ANN

"""
Parametros de la forma del indiSizeviduo (los pesos de las redes).
"""

nUmbrales = 3

nFeatures = 13

nOcultas = 6

nSalida = 3

num_weights = nUmbrales + nFeatures*nOcultas + nOcultas*nSalida

indiSize = {}
indiSize['nUmbrales'] = nUmbrales
indiSize['nFeatures'] = nFeatures
indiSize['nOcultas'] = nOcultas
indiSize['nSalida'] = nSalida

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 88 # Número de indiSizeviduos de la población.
num_parents_mating = 40
num_generations = 55





# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)




for generation in range(num_generations):
    print("Generation : ", generation, "\n")
    # Measing the fitness of each chromosome in the population.
    fitness = M_GA.our_pop_fitness(new_population, indiSize)


    # Selecting the best parents in the population for mating.
    parents = M_GA.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = M_GA.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = M_GA.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # The best result in the current iteration.
    print("Best result (Lowest Error) :   {} %".format( numpy.max(M_GA.our_pop_fitness(new_population, indiSize))))



# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = M_GA.our_pop_fitness(new_population, indiSize)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])

print("Best result (Lowest Error) :   {} %".format( numpy.max(M_GA.our_pop_fitness(new_population, indiSize))))
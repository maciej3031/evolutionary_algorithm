from model import EvolutionaryAlgorithm
import time

# Oszacowanie jak duzo jest trudnych do liczenia funkcji w zbiorze
for func_num in range(1,25+1):
    model = EvolutionaryAlgorithm(population_size=30,
                              number_of_dimensions=10,
                              number_of_parents=15,
                              crossing_likelihood=0.05,
                              mutation_likelihood=0.1,
                              testing_function_number=func_num,
                              crossing_method='average',
                              std_for_mutation_factor=0.01,
                              pair_quality_function='min',
                              max_iterations_number=300,
                              verbose=0)

    time_start = time.time()
    model.run_classic()
    #model.run_marriage()
    print("for {0} time {1}".format(func_num, time.time()-time_start))
    #print(model.scores_history)
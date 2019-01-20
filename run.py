from model import EvolutionaryAlgorithm

model = EvolutionaryAlgorithm(population_size=30,
                              number_of_dimensions=30,
                              number_of_parents=20,
                              crossing_likelihood=0.3,
                              mutation_likehood=0.1,
                              testing_function_number=1,
                              crossing_method='average',
                              std_for_mutation=1,
                              pair_quality_function='min')
# model.run_classic()
model.run_marriage()

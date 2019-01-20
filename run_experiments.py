from model import EvolutionaryAlgorithm
import sys
import numpy as np
import pandas as pd
from cec2005real.cec2005 import Function

#model = EvolutionaryAlgorithm(population_size=30,
#                              number_of_dimensions=30,
#                              number_of_parents=20,
#                              crossing_likelihood=0.3,
#                              mutation_likehood=0.1,
#                              testing_function_number=1,
#                              crossing_method='average',
#                              std_for_mutation=1,
#                              pair_quality_function='min')
# model.run_classic()
# model.run_marriage()
def find_defaults():
    starting_params = dict(population_size=30,
                       number_of_dimensions=30,
                       number_of_parents=20,
                       crossing_likelihood=0.3,
                       mutation_likehood=0.1,
                       testing_function_number=1,
                       crossing_method='average',
                       std_for_mutation_factor=0.01,
                       pair_quality_function='min',
                       max_iterations_number=10) #max, average
    model = EvolutionaryAlgorithm(**starting_params)
    model.run_classic()
    

def main(argv):
    if(len(argv)==0):
        print("No arguments, running all experiments.")
        argv = ['defaults']
    if("defaults" in argv):
        find_defaults()




if __name__ == "__main__":
   main(sys.argv[1:])
from model import EvolutionaryAlgorithm
import sys
import numpy as np
import pandas as pd
from cec2005real.cec2005 import Function

#model = EvolutionaryAlgorithm(population_size=30,
#                              number_of_dimensions=30,
#                              number_of_parents=20,
#                              crossing_likelihood=0.3,
#                              mutation_likelihood=0.1,
#                              testing_function_number=1,
#                              crossing_method='average',
#                              std_for_mutation=1,
#                              pair_quality_function='min')
# model.run_classic()
# model.run_marriage()
    

def find_defaults():
    params = dict(population_size=30,
                  number_of_dimensions=30,
                  number_of_parents=20,
                  crossing_likelihood=0.3,
                  mutation_likelihood=0.1,
                  testing_function_number=1,
                  crossing_method='average',
                  std_for_mutation_factor=0.01,
                  pair_quality_function='min', max_iterations_number=3
                  )
    # Sprawdzane listy
    l_population_size = [6,10,30,50,100]
    l_number_of_parents = [4, 6, 16, 30, 60]
    l_crossing_likelihood = [0.01, 0.1, 0.2, 0.3, 0.4]
    l_crossing_method = ['replace', 'average']
    l_mutation_likelihood = [0.01, 0.1, 0.2, 0.3, 0.4]
    l_std_for_mutation_factor = [0.1, 0.01, 0.005, 0.001, 0.0001]
    #
    l_testing_function_number = [1, 7, 12, 18, 23]
    #number_of_dimensions = 30
    repeats = range(3) 
    def optimize_param(param_name, param_values):
        columns = ["function","trial"]+param_values
        # Wyniki
        results = []
        for testing_function_number in l_testing_function_number:
            for trial in repeats:
                function_results = [testing_function_number, trial]
                for param_value in param_values:
                    params[param_name] = param_value
                    params['testing_function_number'] = testing_function_number
                    model = EvolutionaryAlgorithm(**params)
                    model.run_classic()
                    function_results.append(model.score_for_best_solution)
                results.append(function_results)
        results = pd.DataFrame(columns = columns, data = results)
        results.to_csv("../default_{0}.csv".format(param_name))
        results = results.groupby(by='function').median()
        best_ids = results.drop('trial',axis=1).idxmin(axis=1)
        best_scoring_parameter_value = best_ids.value_counts().idxmax()
        return best_scoring_parameter_value
    # Optymalizacja parametr po parametrze
    params['population_size'] = optimize_param('population_size', l_population_size)
    params['number_of_parents'] = optimize_param('number_of_parents', l_number_of_parents)
    params['crossing_likelihood'] = optimize_param('crossing_likelihood', l_crossing_likelihood)
    params['crossing_method'] = optimize_param('crossing_method', l_crossing_method)
    params['mutation_likelihood'] = optimize_param('mutation_likelihood', l_mutation_likelihood)
    params['std_for_mutation_factor'] = optimize_param('std_for_mutation_factor', l_std_for_mutation_factor)
    # Wyswietlenie wyniku
    print(params)

def main(argv):
    if(len(argv)==0):
        print("No arguments, running all experiments.")
        argv = ['defaults']
    if("defaults" in argv):
        find_defaults()




if __name__ == "__main__":
   main(sys.argv[1:])
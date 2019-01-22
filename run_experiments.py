from model import EvolutionaryAlgorithm
import sys
import numpy as np
import pandas as pd
import time
import multiprocessing
from functools import partial
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
    
def score_param(param_value, params, testing_function_number, param_name):
    exec_time_s = time.time()
    params_copy = params.copy()
    params_copy[param_name] = param_value
    params_copy['testing_function_number'] = testing_function_number
    model = EvolutionaryAlgorithm(**params_copy)
    model.run_classic()
    score = model.score_for_best_solution
    exec_time_s = time.time() - exec_time_s
    print("for {0} = {1}; got: {2};  in time: {3} s".format(param_name, param_value, score, exec_time_s))
    return score,exec_time_s

params = dict(population_size=30,
              number_of_dimensions=10,
              number_of_parents=15,
              crossing_likelihood=0.3,
              mutation_likelihood=0.1,
              testing_function_number=1,
              crossing_method='average',
              std_for_mutation_factor=0.01,
              pair_quality_function='min',
              max_iterations_number=100
              )
# Sprawdzane listy
l_population_size = [6,10,15,30]
                     
l_number_of_parents = [6, 10, 15, 30]
l_crossing_likelihood = [0.05, 0.2, 0.3, 0.4]
l_crossing_method = ['replace', 'average']
l_mutation_likelihood = [0.05, 0.1, 0.2]
l_std_for_mutation_factor = [0.1, 0.01, 0.005, 0.001]
#
l_testing_function_number = [1,7,13,23]
#number_of_dimensions = 30
repeats = range(1) 
def optimize_param(param_name, param_values):        
    columns = ["function","trial"]+param_values + [str(val)+"_time" for val in param_values]
    # Wyniki
    results = []
    for testing_function_number in l_testing_function_number:
        for trial in repeats:
            run_results = [testing_function_number, trial]
            #Rownolegla proba dla roznych wartosci parametru
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            partial_score_param = partial(score_param, 
                                          params = params, 
                                          testing_function_number=testing_function_number, 
                                          param_name=param_name)
            scores_of_params = pool.map(partial_score_param, param_values)
            scores_of_params, times_of_params = zip(*scores_of_params)
            run_results += scores_of_params + times_of_params
            #
            #Sekwencyjna proba dla roznych wartosci parametru
            #for param_value in param_values:
            #    print("testing param: {0} with value: {1} for function nr: {2}".format(
            #            param_name,
            #            param_value,
            #            testing_function_number))
            #    params[param_name] = param_value
            #    params['testing_function_number'] = testing_function_number
            #    model = EvolutionaryAlgorithm(**params)
            #    model.run_classic()
            #    run_results.append(model.score_for_best_solution)
            #
            results.append(run_results)
    results = pd.DataFrame(columns = columns, data = results)
    tested_names = str(param_values)[1:-1].replace(",","_").replace(" ","")
    filename = "../default_{0}_{1}.csv".format(param_name,tested_names)
    print("saving to {0}".format(filename))
    print()
    results.to_csv(filename)
    
    print()
    print("***** Results *****")
    print(results)
    #results = results.groupby(by='function').median()
    #best_ids = results.drop(['trial'],axis=1).idxmin(axis=1)
    #best_scoring_parameter_value = best_ids.value_counts().idxmax()
    #print()
    #print("Parameter {0} was optimized. Choosen: {1}.".format(
    #        param_name, best_scoring_parameter_value))
    #print()
    #return best_scoring_parameter_value

def main(argv):
    # Optymalizacja parametr po parametrze
    optimize_param('population_size', l_population_size)
    #optimize_param('number_of_parents', l_number_of_parents)
    #optimize_param('crossing_likelihood', l_crossing_likelihood)
    #optimize_param('crossing_method', l_crossing_method)
    #optimize_param('mutation_likelihood', l_mutation_likelihood)
    #optimize_param('std_for_mutation_factor', l_std_for_mutation_factor)
    


if __name__ == "__main__":
   main(sys.argv[1:])

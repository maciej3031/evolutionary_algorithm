from model import EvolutionaryAlgorithm
import sys
import numpy as np
import pandas as pd
import time
import multiprocessing
from functools import partial
from cec2005real.cec2005 import Function

repeats_seeds = [5,42,123]
first_function_nr = 1
last_function_nr = 25

def fit_model_classic(seed, params):
    np.random.seed(seed)
    run_time = -time.time()
    print("Running with seed {0}".format(seed))
    model = EvolutionaryAlgorithm(**params)
    model.run_classic()
    run_time += time.time()
    print("Run_seed{0}:\n - took: {1}\n - score: {2}\n - time: {3}".format(seed, 
          run_time,
          model.score_for_best_solution,
          run_time))
    return (model.score_for_best_solution,model.scores_history)

def fit_model_marriage(seed, params):
    np.random.seed(seed)
    run_time = -time.time()
    print("Running with seed {0}".format(seed))
    model = EvolutionaryAlgorithm(**params)
    model.run_marriage()
    run_time += time.time()
    print("Run_seed{0}:\n - took: {1}\n - score: {2}\n - time: {3}".format(seed, 
          run_time,
          model.score_for_best_solution,
          run_time))
    return (model.score_for_best_solution,model.scores_history)

def save_results(results, results_name="../result"):
    # (Wynik, Historia), przy czym Historia = [(iteracja, poprawa), ...]
    print("\nSave results of name: {0}".format(results_name))
    scores, histories = zip(*results)
    print(scores)
    np.save(results_name+"_histories.npy",np.array(histories))
    np.save(results_name+"_scores.npy",np.array(scores))

def run_save_results(function_nr, test_name="Test"):
    params = dict(
              population_size=30,
              number_of_dimensions=10,
              number_of_parents=15,
              crossing_likelihood=0.2,
              mutation_likelihood=0.4,
              crossing_method='average',
              std_for_mutation_factor=0.001,
              verbose = 1,
              testing_function_number=function_nr,
              pair_quality_function='min'
              #,max_iterations_number=10000
              )
    #
    #Problem : Wątki używają tego samego generatora
    # Rozw: podawanie do wątków różnie zainicjowanych macierzy (+ równe szanse alg!)
    #
    run_name = "../{1}_benchmark_f{0}".format(function_nr,test_name)
    print("\n********* Running Function nr {} *********\n".format(function_nr))
    print("*** Run Classic ***")
    pool = multiprocessing.Pool(processes=len(repeats_seeds))
    fit_model = partial(fit_model_classic, params=params)
    results = pool.map(fit_model, repeats_seeds)
    save_results(results,run_name+"_classic")
    print("\n\n*** Run Marriage-Min ***")
    params['pair_quality_function']='min'
    fit_model = partial(fit_model_marriage, params=params)
    results = pool.map(fit_model, repeats_seeds)
    save_results(results,run_name+"_marriage-min")
    print("\n\n*** Run Marriage-Mean ***")
    params['pair_quality_function']='average'
    fit_model = partial(fit_model_marriage, params=params)
    results = pool.map(fit_model, repeats_seeds)
    save_results(results,run_name+"_marriage-mean")
    print("\n\n*** Run Marriage-Max ***")
    params['pair_quality_function']='max'
    fit_model = partial(fit_model_marriage, params=params)
    results = pool.map(fit_model, repeats_seeds)
    save_results(results,run_name+"_marriage-max")

def main(argv):
    current_function_nr = first_function_nr
    while current_function_nr<=last_function_nr:
        run_save_results(current_function_nr, "Test0")
        current_function_nr+=1
    
    

if __name__ == "__main__":
   main(sys.argv[1:])

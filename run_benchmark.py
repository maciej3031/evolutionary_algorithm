from model import EvolutionaryAlgorithm
import sys
import numpy as np
import pandas as pd
import time
import multiprocessing
from functools import partial
from cec2005real.cec2005 import Function

params = dict(
              population_size=30,
              number_of_dimensions=10,
              number_of_parents=15,
              crossing_likelihood=0.2,
              mutation_likelihood=0.05,
              testing_function_number=1,
              crossing_method='average',
              std_for_mutation_factor=0.001
              )

repeats = 3
first_function_nr = 1
last_function_nr = 25

def run_save_results(function_nr, algorithm):
    print("*** Running Function nr {} ***".format(function_nr))


def main(argv):
    current_function_nr = first_function_nr
    while current_function_nr<=last_function_nr:
        run_save_results(current_function_nr)
        current_function_nr+=1

if __name__ == "__main__":
   main(sys.argv[1:])

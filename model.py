import numpy as np
from cec2005real.cec2005 import Function
from scipy.special import softmax
from tqdm import tqdm


class EvolutionaryAlgorithm:
    def __init__(self, population_size, number_of_dimensions, number_of_parents, crossing_likelihood,
                 testing_function_number, crossing_method='average', std_for_mutation=1, pair_quality_function='min'):
        self.population_size = population_size
        self.number_of_dimensions = number_of_dimensions
        self.number_of_parents = number_of_parents
        self.crossing_likelihood = crossing_likelihood
        self.testing_function_number = testing_function_number
        self.testing_function_object = Function(testing_function_number, number_of_dimensions)
        self.testing_function = self.testing_function_object.get_eval_function()
        self.bottom_limit = self.testing_function_object.info()['lower']
        self.top_limit = self.testing_function_object.info()['upper']
        self.threshold = self.testing_function_object.info()['threshold']
        self.crossing_method = crossing_method
        self.std_for_mutation = std_for_mutation
        self.max_iterations_number = 10000 * population_size
        self.best_solution_found = None
        self.score_for_best_solution = 999999999999999
        self.pair_quality_function = pair_quality_function

    def _cross(self, parents):
        children = []
        for parent in parents:
            for other_parent in parents:
                if any(parent != other_parent):
                    child = self._single_cross(parent, other_parent)
                    if any(parent != child):
                        children.append(child)
        return np.array(children)

    def _single_cross(self, parent_1, parent_2):
        child = parent_1.copy()
        if self.crossing_method == 'replace':
            if np.random.rand() < self.crossing_likelihood:
                dimensions_indices = np.array(range(self.number_of_dimensions))
                chosen_index = np.random.choice(dimensions_indices)
                child[chosen_index:] = parent_2[chosen_index:]

        elif self.crossing_method == 'average':
            if np.random.rand() < self.crossing_likelihood:
                child = (parent_1 + parent_2) / 2

        else:
            raise Exception('No such crossing method as "{}"!'.format(self.crossing_method))

        return child

    def _mutate(self, children):
        return np.random.normal(children, self.std_for_mutation)

    def _get_pair_quality(self, pair):
        score_1 = self.testing_function(pair[0])
        score_2 = self.testing_function(pair[1])

        if self.pair_quality_function == 'average':
            return (score_1 + score_2) / 2
        elif self.pair_quality_function == 'min':
            return min([score_1, score_2])
        elif self.pair_quality_function == 'max':
            return max([score_1, score_2])
        else:
            raise Exception('No such pair quality function as "{}"!'.format(self.pair_quality_function))

    def show_testing_function(self):
        print(self.testing_function_object.info(), end='\n')

    def run_classic(self):
        self.show_testing_function()
        # 1. Inicjacja populacji
        population = np.random.uniform(self.bottom_limit, self.top_limit,
                                       (self.population_size, self.number_of_dimensions))
        for n in tqdm(range(self.max_iterations_number)):
            # 2. Ocena osobników
            scores = []
            for individual in population:
                score = self.testing_function(individual)
                scores.append(score)

            worst_value = max(scores)
            best_value = min(scores)

            normalized_inverted_scores = [(1 - (rate - best_value) / (worst_value - best_value)) for rate in scores]
            probs = softmax(normalized_inverted_scores)

            # 3. Wybór osobników
            population_indices = np.array(range(self.population_size))

            chosen_indices = np.random.choice(population_indices, self.number_of_parents, p=probs)
            parents = population[chosen_indices]

            # 4. Krzyżowanie i mutacja wybranych osobników z uzyskaniem zbioru potomków
            children = self._cross(parents)
            children = self._mutate(children)

            # 5. Nowa populacja z najlepszych
            if children.size > 0:
                children_and_parents = np.concatenate([population, children], axis=0)
            else:
                children_and_parents = population

            if self.testing_function_number not in [7, 25]:
                children_and_parents = np.clip(children_and_parents, self.bottom_limit, self.top_limit)

            scores = []
            for individual in children_and_parents:
                scores.append((score, individual))
                score = self.testing_function(individual)

            scores = sorted(scores, key=lambda x: x[0])
            best_scores = scores[:self.population_size]

            population = np.array([item[1] for item in best_scores])

            # 6. Wybierz najlepszy i zapisz do self.best_solution_found
            if self.score_for_best_solution > best_scores[0][0]:
                self.best_solution_found = best_scores[0][1]
                self.score_for_best_solution = best_scores[0][0]
                print('New best score: ', self.score_for_best_solution)

            # 7. Przerwij jeżeli osiągneliśmy minimum
            if self.score_for_best_solution - self.threshold <= 0:
                print('Solution found after {} iteration'.format(n))
                break

        print("Solution: ", self.best_solution_found)
        print("Score for solution :", self.score_for_best_solution)

    def run_marriage(self):
        self.show_testing_function()
        # 1. Inicjacja populacji
        population = np.random.uniform(self.bottom_limit, self.top_limit,
                                       (self.population_size, self.number_of_dimensions))
        # 1a. Dobranie osobników w pary
        np.random.shuffle(population)
        pairs = np.stack([population[i:i + 2] for i in range(0, len(population), 2)])

        for n in tqdm(range(self.max_iterations_number)):
            # 2. Ocena par
            scores = []
            for pair in pairs:
                score = self._get_pair_quality(pair)
                scores.append(score)

            worst_value = max(scores)
            best_value = min(scores)

            normalized_inverted_scores = [(1 - (rate - best_value) / (worst_value - best_value)) for rate in scores]
            probs = softmax(normalized_inverted_scores)

            # 3. Wybór najlepszych par
            pairs_indices = np.array(range(self.population_size // 2))
            chosen_indices = np.random.choice(pairs_indices, self.number_of_parents // 2, p=probs)
            parents_pairs = pairs[chosen_indices]
            parents = parents_pairs.reshape(parents_pairs.shape[0] * parents_pairs.shape[1], parents_pairs.shape[2])

            # 4. Krzyżowanie i mutacja wybranych par z uzyskaniem zbioru potomków
            children = self._cross(parents)
            children = self._mutate(children)

            # 4a. Dobranie dzieci w pary i ocena
            if children.shape[0] % 2:
                children = children[:-1]

            if children.size > 1:
                np.random.shuffle(children)
                children_pairs = np.stack([children[i:i + 2] for i in range(0, len(children), 2)])

            # 5. Nowa populacja z najlepszych
            if children.size > 0:
                children_and_parents_pairs = np.concatenate([pairs, children_pairs], axis=0)
            else:
                children_and_parents_pairs = pairs

            if self.testing_function_number not in [7, 25]:
                children_and_parents_pairs = np.clip(children_and_parents_pairs, self.bottom_limit, self.top_limit)

            scores = []
            for pair in children_and_parents_pairs:
                score = self._get_pair_quality(pair)
                scores.append((score, pair))

            scores = sorted(scores, key=lambda x: x[0])
            best_scores = scores[:self.population_size // 2]

            pairs = np.array([item[1] for item in best_scores])

            # 6. Wybierz najlepszą parę i zapisz do self.best_solution_found
            if self.score_for_best_solution > best_scores[0][0]:
                self.best_solution_found = best_scores[0][1]
                self.score_for_best_solution = best_scores[0][0]
                print('New best score: ', self.score_for_best_solution)

            # 7. Przerwij jeżeli osiągneliśmy minimum
            if self.score_for_best_solution - self.threshold <= 0:
                print('Solution found after {} iteration'.format(n))
                break

        print("Solution: ", self.best_solution_found)
        print("Score for solution :", self.score_for_best_solution)

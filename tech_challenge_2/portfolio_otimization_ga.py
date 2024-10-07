import random
import copy
import numpy as np
import yfinance as yf


class PortfolioOptimizationGA:

    MUTATION_PROBABILITY = 0.3

    def __init__(self, assets, population_size, n_generations, period_historical="1y",):
        if len(assets) < 2:
            raise ValueError("At least two assets are required.")

        self.assets = assets
        self.population_size = population_size
        self.n_generations = n_generations
        self.period_historical = period_historical

    def get_historical_assets(self):
        for asset in self.assets:
            yf_asset = yf.Ticker(asset)
            yf_asset_hist = yf_asset.history(period=self.period_historical)
            yf_asset_hist['Daily_Return'] = yf_asset_hist['Close'].pct_change() * 100

            yf_asset_mean = yf_asset_hist['Daily_Return'].mean()
            yf_asset_std = yf_asset_hist['Daily_Return'].std()

    def generate_random_individual(self, generation, i):
        rng1 = np.random.default_rng([generation, i])
        random_weights = np.array(rng1.random(len(self.assets)))
        rebalanced_weights = random_weights / np.sum(random_weights)
        return rebalanced_weights

    def generate_random_population(self, generation):
        return [self.generate_random_individual(generation, i) for i in range(self.population_size)]

    def generate_new_individual(self, parent_1, parent_2):
        # Crossover
        individual = self.crossover(parent_1, parent_2)
        # Mutation
        individual = self.mutate(individual)

        return individual

    def generate_new_population(self, population):
        new_population = [population[0]]  # Keep the best individual: ELITISM

        while len(new_population) < self.population_size:
            # Select parents
            best_parents = 10
            parent_1, parent_2 = random.choices(population[:best_parents], k=2)

            # Generate new individual
            individual = self.generate_new_individual(parent_1, parent_2)
            new_population.append(individual)

        return new_population

    def calculate_fitness(self, individual):
        sharpe_ratio = 0.0
        return sharpe_ratio

    def sort_population(self, population, population_fitness):
        # Combine lists into pairs
        combined_lists = list(zip(population, population_fitness))

        # Sort based on the values of the fitness list
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1], reverse=True)

        # Separate the sorted pairs back into individual lists
        sorted_population, sorted_population_fitness = zip(*sorted_combined_lists)

        return sorted_population, sorted_population_fitness

    def crossover(self, parent_1, parent_2):
        individual_1 = np.zeros(len(self.assets))

        if len(parent_1) < 3:
            return individual_1

        individual_1 = (parent_1 + parent_2) / 2

        return individual_1

    def mutate(self, individual):
        mutated_individual = copy.deepcopy(individual)

        n = random.random()
        if n < self.MUTATION_PROBABILITY:
            # Individual must have 2 assets to perform mutation
            if len(individual) < 2:
                return individual

            index = random.randint(0, len(individual) - 2)

            mutated_individual[index], mutated_individual[index + 1] = mutated_individual[index + 1], mutated_individual[index]

        return mutated_individual

    def process_generation(self, generation, population):
        print(f"Processing Generation {generation} of total {self.n_generations}")
        population_fitness = [self.calculate_fitness(individual) for individual in population]

        population, population_fitness = self.sort_population(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        return best_solution, best_fitness

    def process(self):
        best_fitness_values = []
        best_solutions = []

        generation = 0
        population = self.generate_random_population(generation)
        best_solution, best_fitness = self.process_generation(generation, population)
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        for generation in range(1, self.n_generations):
            print(f"Generating new population at generation {generation}")
            population = self.generate_new_population(population)

            best_solution, best_fitness = self.process_generation(generation, population)

            print(f"Generation {generation}: Best fitness = {best_fitness} / Best Solution = {best_solution}")

            best_fitness_values.append(best_fitness)
            best_solutions.append(best_solution)




if __name__ == "__main__":
    assets = ['PETR4.SA', 'ITUB4.SA', '^BVSP']
    portfolio_optimization_ga = PortfolioOptimizationGA(assets, 100, 100, period_historical="2y")
    # portfolio_optimization_ga.get_historical_assets()
    portfolio_optimization_ga.process()
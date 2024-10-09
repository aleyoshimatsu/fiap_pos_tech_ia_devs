import random
import copy
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class PortfolioOptimizationGA:

    def __init__(self, benchmark, assets, population_size, n_generations, mutation_probability=0.3):
        if len(assets) < 2:
            raise ValueError("At least two assets are required.")

        self.benchmark = benchmark
        self.assets = assets
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_probability = mutation_probability
        self.df_benchmark = pd.DataFrame()
        self.df_assets = pd.DataFrame()
        self.best_fitness_values = []
        self.best_solutions = []

    def get_historical_assets(self, start_date, end_date):
        tickers = " ".join(self.assets)

        data = yf.download(tickers, start=start_date, end=end_date)
        data_pct_change = data['Adj Close'].pct_change()
        data_pct_change = data_pct_change.dropna()
        self.df_assets = data_pct_change.copy()

        data = yf.download(self.benchmark, start=start_date, end=end_date)
        data_pct_change = data['Adj Close'].pct_change()
        data_pct_change = data_pct_change.dropna()
        self.df_benchmark = data_pct_change.copy()

    def generate_random_individual(self, generation, i):
        # rng1 = np.random.default_rng([generation, i])
        random_weights = np.array(np.random.random(len(self.assets)))
        rebalanced_weights = random_weights / np.sum(random_weights)
        rebalanced_weights = np.around(rebalanced_weights, decimals=2)
        return rebalanced_weights

    def generate_random_population(self, generation):
        return [self.generate_random_individual(generation, i) for i in range(self.population_size)]

    def generate_new_individual(self, parent_1, parent_2):
        # Crossover
        individual_1, individual_2 = self.crossover(parent_1, parent_2)
        # Mutation
        individual_1 = self.mutate(individual_1)
        individual_2 = self.mutate(individual_2)

        return individual_1, individual_2

    def generate_new_population(self, population):
        new_population = [population[0]]  # ELITISM

        while len(new_population) < self.population_size:
            # Select parents
            best_parents = 10
            parent_1, parent_2 = random.choices(population[:best_parents], k=2)

            # Generate new individuals
            individual_1, individual_2 = self.generate_new_individual(parent_1, parent_2)
            new_population.append(individual_1)
            new_population.append(individual_2)

        return new_population

    def calculate_fitness(self, individual):
        p = np.asmatrix(self.df_assets.mean())
        w = np.asmatrix(individual)
        c = np.asmatrix(self.df_assets.cov())

        mu = w * p.T # Retorno Esperado da Carteira
        sigma = np.sqrt(w * c * w.T) # Desvio Padrão da Carteira

        sharpe_ratio = mu / sigma

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
        crossover_point = random.randint(1, len(parent_1) - 1)
        # Ensure that the slices have the same size
        child1 = np.concatenate(
            (parent_1[:crossover_point], parent_2[crossover_point:]))
        child2 = np.concatenate(
            (parent_2[:crossover_point], parent_1[crossover_point:]))

        if np.sum(child1) < 1:
            child1[child1.argmin()] = 1 - np.sum(child1) + child1[child1.argmin()]
        if np.sum(child2) < 1:
            child2[child2.argmin()] = 1 - np.sum(child2) + child2[child2.argmin()]

        return child1, child2

    def mutate(self, individual):
        mutated_individual = copy.deepcopy(individual)

        n = random.random()
        if n < self.mutation_probability:
            # Individual must have 2 assets to perform mutation
            if len(individual) < 2:
                return individual

            index = random.randint(0, len(individual) - 2)

            mutated_individual[index], mutated_individual[index + 1] = mutated_individual[index + 1], mutated_individual[index]
            mutated_individual = np.around(mutated_individual, decimals=2)

        return mutated_individual

    def process_generation(self, generation, population):
        print(f"Processing Generation {generation} of total {self.n_generations}")
        population_fitness = [self.calculate_fitness(individual) for individual in population]

        population, population_fitness = self.sort_population(population, population_fitness)

        return population, population_fitness

    def process(self):
        generation = 0
        population = self.generate_random_population(generation)
        population, population_fitness = self.process_generation(generation, population)
        self.best_fitness_values.append(population_fitness[0])
        self.best_solutions.append(population[0])

        for generation in range(1, self.n_generations):
            print(f"Generating new population at generation {generation}")
            population = self.generate_new_population(population)

            population, population_fitness = self.process_generation(generation, population)

            print(f"Generation {generation}: Best fitness = {population_fitness[0]} / Best Solution = {population[0]}")

            self.best_fitness_values.append(population_fitness[0])
            self.best_solutions.append(population[0])

    def plot_results(self):
        portfolio_returns = (self.df_assets
                             * self.best_solutions[-1]).sum(axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_returns.cumsum(), label=f'Optimized Portfolio {self.best_solutions[-1]}')
        plt.plot(self.df_benchmark.cumsum(), label=f'Benchmark {self.benchmark}')  # Example benchmark
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title(f'Backtesting Results {self.assets}')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    start_date = '2020-01-01'
    end_date = '2024-10-09'

    bench = "^BVSP"
    assets = ['BPAC11.SA', 'ITUB4.SA', 'BBAS3.SA', 'BPAN4.SA', 'BBDC4.SA']
    # Best fitness = [[0.03155492]] / Best Solution = [0.25 0.02 0.54 0.07 0.12]
    # Best fitness = [[0.03208329]] / Best Solution = [0.28 0.01 0.57 0.12 0.02]
    # Best fitness = [[0.03129325]] / Best Solution = [0.45 0.02 0.48 0.06 0.06]
    # Best fitness = [[0.03084563]] / Best Solution = [0.22 0.   0.5  0.12 0.24]

    portfolio_optimization_ga = PortfolioOptimizationGA(bench, assets, 100, 1000)
    portfolio_optimization_ga.get_historical_assets(start_date, end_date)
    portfolio_optimization_ga.process()
    portfolio_optimization_ga.plot_results()
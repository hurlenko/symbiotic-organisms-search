import numpy as np


class SOS(object):
    def __init__(self,
                 l_bound,
                 u_bound,
                 population_size,
                 fitness_vector_size):
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.population_size = population_size
        self.fitness_vector_size = fitness_vector_size
        self.population = None
        self.best = None

    def float_rand(self, a, b, size=None):
        return a + ((b - a) * np.random.random(size))

    def generate_population(self):
        population = self.float_rand(self.l_bound, self.u_bound, (self.population_size, self.fitness_vector_size))
        self.population = np.array([Individual(p) for p in population])
        self.best = sorted(self.population, key=lambda x: x.fitness)[0]

    def mutualism(self, a_index):
        b_index = np.random.permutation(np.delete(np.arange(self.population_size), a_index))[0]
        b = self.population[b_index]
        a = self.population[a_index]
        bf1, bf2 = np.random.randint(1, 3, 2)  # benefit factor1
        array_rand = np.random.random(self.fitness_vector_size)
        mutual = (a.phenotypes + b.phenotypes) / 2
        new_a = a.phenotypes + (array_rand * (self.best.phenotypes - (mutual * bf1)))
        new_b = b.phenotypes + (array_rand * (self.best.phenotypes - (mutual * bf2)))
        new_a = Individual([self.u_bound if x > self.u_bound
                            else self.l_bound if x < self.l_bound else x for x in new_a])
        new_b = Individual([self.u_bound if x > self.u_bound
                            else self.l_bound if x < self.l_bound else x for x in new_b])
        self.population[a_index] = new_a if new_a.fitness < a.fitness else a
        self.population[b_index] = new_b if new_b.fitness < b.fitness else b

    def commensalism(self, a_index):  
        b_index = np.random.permutation(np.delete(np.arange(len(self.population)), a_index))[0]
        b = self.population[b_index]
        a = self.population[a_index]
        array_rand = self.float_rand(-1, 1, self.fitness_vector_size)
        new_a = a.phenotypes + (array_rand * (self.best.phenotypes - b.phenotypes))
        new_a = Individual([self.u_bound if x > self.u_bound
                            else self.l_bound if x < self.l_bound
                            else x for x in new_a])
        self.population[a_index] = new_a if new_a.fitness <= a.fitness else a

    def parasitism(self, a_index):
        parasite = np.array(self.population[a_index].phenotypes)
        b_index = np.random.permutation(np.delete(np.arange(len(self.population)), a_index))[0]
        b = self.population[b_index]
        parasite[np.random.randint(0, self.fitness_vector_size)] = self.float_rand(self.l_bound, self.u_bound)
        parasite = Individual(parasite)
        self.population[b_index] = parasite if parasite.fitness <= b.fitness else b

    def proceed(self, steps):
        for j in range(1, steps + 1):
            for i, val in enumerate(self.population):
                self.mutualism(i)
                self.commensalism(i)
                self.parasitism(i)
                self.best = sorted(self.population, key=lambda x: x.fitness)[0]
            if j % 50 == 0:
                print('{0}/{1} Current population:'.format(j, steps))
                print(self.best)


class Individual(object):
    def __init__(self, phenotypes):
        self.phenotypes = np.array(phenotypes)  # phenotype
        self.fitness = fitness_func(self.phenotypes)  # value of the fitness function

    def __str__(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)


def fitness_func(arg_vec):
    # Sphere model (DeJong1)
    # return np.sum([x ** 2 for x in arg_vec])
    # Rosenbrock's saddle (DeJong2)
    # return sum([(100 * (xj - xi ** 2) ** 2 + (xi - 1) ** 2) for xi, xj in zip(arg_vec[:-1], arg_vec[1:])])
    # Rastrigin's function
    # return 10 * len(arg_vec) + np.sum([x ** 2 - 10 * np.cos(2 * np.pi * x) for x in arg_vec])
    # Ackley's Function
    # s1 = -0.2 * np.sqrt(np.sum([x ** 2 for x in arg_vec]) / len(arg_vec))
    # s2 = np.sum([np.cos(2 * np.pi * x) for x in arg_vec]) / len(arg_vec)
    # return 20 + np.e - 20 * np.exp(s1) - np.exp(s2)
    # SN
    s1 = (sum(arg_vec) - sum(x*x for x in arg_vec)) * sum(np.cos(x) for x in arg_vec)
    s2 = 4 / (np.sqrt(np.abs(np.tan(sum(arg_vec))))) + int(sum(x*x for x in arg_vec))
    return s1 / s2


interval = (-1, 1)
pop_size = 50  # The number of candidate solutions
max_iter = 1000  # The number of iterations
dim = 5  # number of problem variables


def main():
    sos = SOS(interval[0], interval[1], pop_size, dim)
    sos.generate_population()
    print('Initial population')
    for ind in sorted(sos.population, key=lambda x: x.fitness):
        print(ind)
    sos.proceed(max_iter)

if __name__ == '__main__':
    main()

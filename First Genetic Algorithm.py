import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

num_bikes = 150

#bike_id = list(range(num_bikes))

test_df = pd.DataFrame([[0,1], [1,1], [1,0], [0,0.5], [0,0]])
test_dist_m = distance_matrix(test_df, test_df)

df = pd.DataFrame({'x':np.random.random(size = num_bikes),
                   'y':np.random.random(size = num_bikes)})

shop = pd.DataFrame([[0,0]], columns = ['x', 'y'])
df = df.append(shop)

dist_m = distance_matrix(df, df)
max_dist = 15

class Algorithm():
    def __init__(self, parents):
        
        self.mutations = []
        self.insertions = []
        
        if not parents:
            self.random_generation()
        else:
            self.parents = parents
            self.solution = self.recombination(parents)
            
            if self.solution is None:
                self.random_generation()        
            else:
                self.mutation()
                self.solution_size = len(self.solution) - 2
        
        self.new_gene()
    
    def random_generation(self):
        self.parents = []
        self.solution_size = np.random.randint(low = 0, high = num_bikes)
        
        self.solution = np.random.choice(list(range(1, num_bikes+1)), size = self.solution_size, replace = False)
        self.solution = np.insert(self.solution, obj = 0, values = 0)
        self.solution = np.insert(self.solution, obj = len(self.solution), values = 0)
        
    def recombination(self, parents):
        
        max_trials = 100
        num_trials = 0
        
        while True:
            self.recom_spot = np.random.randint(low=1, high=len(parents[0].solution))
            self.first_bit = parents[0].solution[:self.recom_spot]
            self.second_bit = parents[1].solution[self.recom_spot:]
        
            new_sol = np.insert(self.first_bit, obj = self.recom_spot, values = self.second_bit)
            num_trials += 1
        
            if len(new_sol) == len(set(new_sol)) + 1: # success - no duplicate values except 0
                return new_sol
            if num_trials == max_trials:
                return
        
    def mutation(self):
        mutation_rate = 0.2
        
        choices = list(set(self.solution).symmetric_difference(set(list(range(num_bikes))))) # which sites are not in the solution
        
        for i in range(1, len(self.solution) - 1): # can't mutate first or last bit
            if (np.random.uniform() <= mutation_rate) and (len(choices) != 0):
                new_gene = np.random.choice(choices)
                
                self.mutations.append([self.solution[i], new_gene])
                self.solution[i] = new_gene

    def new_gene(self):
        new_gene_rate = 10/len(self.solution) 
        choices = list(set(self.solution).symmetric_difference(set(list(range(num_bikes)))))

        if (np.random.uniform() <= new_gene_rate) and (len(choices) != 0):

            new_gene_loc = np.random.randint(0, len(self.solution))
            new_gene = np.random.choice(choices)
            
            self.insertions.append([new_gene_loc, new_gene])
            self.solution = np.insert(self.solution, obj = new_gene_loc, values = new_gene)
    
    def get_distance(self, dist_matrix):
        self.dist = 0
        
        for i in range(0, len(self.solution) - 1):
            current_loc = self.solution[i]
            next_loc = self.solution[i+1]
            
            self.dist = self.dist + dist_matrix[current_loc, next_loc]
            current_loc = self.solution[i]
        
        return self.dist

    def get_fitness(self):
        return self.solution_size

def run_simulation(pop_size, distance_matrix, num_iter):
    global algs
    algs = []

    num_reproducing = int(np.floor(pop_size/2))
    
    initialize(pop_size, distance_matrix)
    summarize_generations(algs, dist_m, 0)
    for i in range(num_iter):

        algs = find_parents(algs, num_reproducing)
        algs = next_generation(algs, dist_m)
        summarize_generations(algs, dist_m, i)


def initialize(pop_size, distance_matrix):
    while len(algs) < pop_size:
        new_alg = Algorithm(parents = False)
        
        if new_alg.get_distance(distance_matrix) <= max_dist: # feasible solution
            algs.append(new_alg)

def find_parents(algs, num_reproducing):
    max_val = sorted([algs.get_fitness() for algs in algs], reverse = True)[num_reproducing-1]
    potential_parents = [algs for algs in algs if algs.get_fitness() >= max_val]
    
    return potential_parents

def next_generation(algs, distance_matrix):
    pot_par = algs[:]

    total_fitness = sum([p.get_fitness() for p in pot_par])
    repr_prob = [p.get_fitness()/total_fitness for p in pot_par]
    
    while len(algs) < pop_size:
        first_par = np.random.choice(pot_par, p = repr_prob)
        second_par = np.random.choice(pot_par, p = repr_prob)

        while first_par == second_par:
            second_par = np.random.choice(pot_par, p = repr_prob)
            
        new_alg = Algorithm(parents = [first_par, second_par])
        
        if new_alg.get_distance(distance_matrix) <= max_dist: # feasible solution
            algs.append(new_alg)

    return algs


def summarize_generations(algs, distance_matrix, i):
    num_algs = len(algs)
    total_fitness = sum([a.get_fitness() for a in algs])
    total_distance = sum(a.get_distance(distance_matrix) for a in algs)
    
    avg_fitness = round(total_fitness/num_algs, 3)
    avg_distance = round(total_distance/num_algs, 3)

    print("{}\t{}\t{}".format(i, avg_fitness, avg_distance))


run_simulation(100, dist_m, 20)

pot_par = algs[:2]
new = Algorithm(parents = pot_par)
new.get_distance(dist_m)
new.get_fitness()
print(new.solution.tolist())
pot_par[0].solution[:new.recom_spot].tolist(), pot_par[1].solution[new.recom_spot:].tolist()


algs[0].get_distance(dist_m)
algs[0].solution.tolist()
algs[1].get_distance(dist_m)
algs[1].solution.tolist()
df[list(algs[1].solution), ]
my_df = df[df.iloc(list(algs[1].solution))]



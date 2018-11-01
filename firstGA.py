import numpy as np

num_bikes = 150
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
    
    def random_generation(self, solution_size = None):
        self.parents = []
        
        if solution_size is not None:
            self.solution_size = solution_size
        else:
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
        
    def mutation(self, mutation_rate = 0.2):
        # which sites are not in the solution        
        choices = list(set(self.solution).symmetric_difference(set(list(range(num_bikes))))) 

        for i in range(1, len(self.solution) - 1): # can't mutate first or last bit
            if (np.random.uniform() <= mutation_rate) and (len(choices) != 0):
                new_gene = np.random.choice(choices)
                self.mutations.append([self.solution[i], new_gene])
                self.solution[i] = new_gene

    def new_gene(self, new_gene_rate = None):
        
        if new_gene_rate is None:
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

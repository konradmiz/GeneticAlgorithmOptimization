import numpy as np
import pandas as pd
import uuid

df = pd.read_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/Data/df.csv")
dist_m = np.loadtxt(open("C:/Users/Konrad/Desktop/GeneticAlgorithm/Data/dist.csv", "rb"), delimiter=",", skiprows=1)
#pd.read_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/dist.csv")
num_bikes = df.shape[0] - 1
bike_dist = np.loadtxt(open("C:/Users/Konrad/Desktop/GeneticAlgorithm/Data/workshopdist.csv", "rb"), delimiter=",", skiprows=1)
bike_dist = (max(bike_dist) - bike_dist + 1) ** 10
bike_prob = bike_dist/bike_dist.sum()

bike_prob /= bike_prob.sum()
#sum(bike_prob)
class Algorithm():
    def __init__(self, parents, i, bike_prob = bike_prob):
        
        self.id = str(uuid.uuid1())
        self.trial_no = i
        
        self.mutations = []
        self.swaps = []
        self.insertions = []
        self.inversions = []
        
        if not parents:
            self.random_generation(bike_prob)
        else:
            self.parents = parents
            self.solution = self.recombination(parents)
            
            if self.solution is None:
                self.random_generation(bike_prob)        
            else:
                self.mutation()
                self.inversion()
                self.solution_size = len(self.solution) - 2
        
        self.new_gene()
    
    def random_generation(self, bike_prob = bike_prob):
        self.parents = []
        
        #if solution_size is not None:
            #self.solution_size = solution_size
        #else:
        self.solution_size = np.random.randint(low = 0, high = num_bikes+1)
        self.solution = np.random.choice(list(range(1, num_bikes+1)), p = bike_prob, size = self.solution_size, replace = False)
        self.solution = np.insert(self.solution, obj = 0, values = 0)
        self.solution = np.insert(self.solution, obj = len(self.solution), values = 0)
       
    def recombination(self, parents):
        max_trials = 50
        num_trials = 0
        
        while True:
            if (len(parents[0].solution) <= 2):
                return
            
            self.recom_spot = np.random.randint(low=1, high=len(parents[0].solution)-1)
            self.first_bit = parents[0].solution[:self.recom_spot]
            self.second_bit = parents[1].solution[self.recom_spot:]
        
            new_sol = np.insert(self.first_bit, obj = self.recom_spot, values = self.second_bit)
            num_trials += 1
        
            if len(new_sol) == len(set(new_sol)) + 1: # success - no duplicate values except 0
                self.recom = 1
                return new_sol
            if num_trials == max_trials:
                return
            
    def inversion(self, inversion_rate = 0.1):
        if np.random.uniform() <= inversion_rate:
            if len(self.solution) <= 2:
                return
            
            inversion_pt = np.random.choice(list(range(1, len(self.solution) - 1)))
            
            self.first_part = self.solution[:inversion_pt]
            self.second_part = np.flipud(self.solution[inversion_pt:len(self.solution)-1])
            self.second_part = np.concatenate((self.second_part, [0]))
            # keep the solution until the recombination point, invert the remaining bits
            
            self.solution = np.concatenate((self.first_part, self.second_part))
            self.inversions.append([self.first_part, self.second_part])
        
    def mutation(self, mutation_rate = 0.5):
        for i in range(1, len(self.solution) - 1): # can't mutate first or last bit
             # which sites are not in the solution             
            if (np.random.uniform() <= mutation_rate):
                choices = list(set(list(range(1, num_bikes+1))) - set(self.solution))
                
                if (len(choices) > 0) & (np.random.random() <= 0.5):
                    
                    extra_dist = np.zeros(shape = len(choices))

                    for j, b in enumerate(choices):
                        extra_dist[j] = dist_m[self.solution[i-1], choices[j]] + \
                                        dist_m[choices[j], self.solution[i]] - \
                                        dist_m[self.solution[i-1], self.solution[i]]
                                        
                    extra_dist_magnified = (max(extra_dist) - extra_dist + 1) ** 10
                    extra_dist_final = extra_dist_magnified/extra_dist_magnified.sum()
                    
                    new_allele = np.random.choice(choices, p = extra_dist_final)
                    self.mutations.append([i, self.solution[i], new_allele])
                    self.solution[i] = new_allele
                else:
                    swap_idx_1 = np.random.choice(list(range(1, len(self.solution) - 1)))
                    swap_idx_2 = np.random.choice(list(range(1, len(self.solution) - 1)))
                    
                    gene_1 = self.solution[swap_idx_1]
                    gene_2 = self.solution[swap_idx_2]
                    
                    self.solution[swap_idx_1] = gene_2
                    self.solution[swap_idx_2] = gene_1
                    self.swaps.append([gene_1, gene_2])
            
    def new_gene(self, new_gene_rate = None):

        if new_gene_rate is None:
            new_gene_rate = 0.08
            #0.5/len(self.solution)

        choices = list(set(list(range(1, num_bikes+1))) - set(self.solution))
        
        if (np.random.uniform() <= new_gene_rate) and (len(choices) > 0):
            new_gene_loc = np.random.randint(1, len(self.solution)) # can't insert into first or last
 
            extra_dist = np.zeros(shape = len(choices))

            for i, b in enumerate(choices):
                #print(i,b)
                extra_dist[i] = dist_m[self.solution[new_gene_loc-1], choices[i]] + \
                                dist_m[choices[i], self.solution[new_gene_loc]] - \
                                dist_m[self.solution[new_gene_loc-1], self.solution[new_gene_loc]]
                                
            extra_dist_magnified = (max(extra_dist) - extra_dist + 1) ** 10
            extra_dist_final = extra_dist_magnified/extra_dist_magnified.sum()
            
            new_gene = np.random.choice(choices, p = extra_dist_final)
 
            self.insertions.append([(new_gene_loc, new_gene)])
            self.solution = np.insert(self.solution, obj = new_gene_loc, values = new_gene)
            self.solution_size += 1
    
    def get_distance(self, dist_matrix):
        self.dist = 0
        
        for i in range(0, len(self.solution) - 1):
            current_loc = self.solution[i]
            next_loc = self.solution[i+1]
            
            #self.dist = self.dist + dist_matrix[(dist_matrix['from'] == current_loc) & (dist_matrix['to'] == next_loc)].dist.item()
            self.dist = self.dist + dist_matrix[current_loc, next_loc] + 500
        
        return self.dist

    def get_fitness(self):
        return self.solution_size

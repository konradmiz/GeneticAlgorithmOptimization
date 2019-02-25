import numpy as np
import Environment

class Algorithm():
    """ Solution object of GA type
    
    GAs are initialized either through the recombination of the genotypes of
    the two parents, or through a random solution selection process. The solutions
    are mutated, inverted, and new genes may be added to force longer and 
    longer solutions over time. 
    
    """
    
    def __init__(self, parents, i, bike_prob = Environment.BIKE_PROB):
        """ Creates a new genetic algorithm solution
        
        If there are no parents, generate the solution randomly. Otherwise,
        recombine the solutions of the parents and mutate and invert the genes
        
        """
        self.trial_no = i
        self.recom = 0

        self.mutations = []
        self.swaps = []
        self.insertions = []
        self.inversions = []
        
        if not parents:
            self.random_generation(Environment.BIKE_PROB)
        else:
            self.parents = parents
            self.solution = self.recombination(parents)
            
            if self.solution is None:
                self.random_generation(Environment.BIKE_PROB)        
            else:
                self.mutation()
                self.inversion()
                self.solution_size = len(self.solution) - 2
        
        self.new_gene()
    
    def random_generation(self, bike_prob = Environment.BIKE_PROB):
        """ Create a random solution chromosome
        
        If there are no parents (first generation or recombination didn't work),
        create a new solution of random length. 
        
        """
        
        self.parents = []
        
        self.solution_size = np.random.randint(low = 0, high = Environment.NUM_BIKES + 1)
        self.solution = np.random.choice(list(range(1, Environment.NUM_BIKES + 1)), 
                                         p = bike_prob, size = self.solution_size, replace = False)
        
        self.solution = np.insert(self.solution, obj = 0, values = 0)
        self.solution = np.insert(self.solution, obj = len(self.solution), values = 0)
       
    def recombination(self, parents):
        """ Two-point crossover
        
        Try to recombine the chromosomes of parents. Double recombination used.
        """
        
        max_trials = 4
        num_trials = 0
        
        while True:
            if (len(parents[0].solution) <= 2):
                return
            
            self.recom_spot_one = np.random.randint(low=1, high=len(parents[0].solution)-1)
            self.recom_spot_two = np.random.randint(low=self.recom_spot_one, high=len(parents[0].solution)-1)
            
            self.first_bit = parents[0].solution[:self.recom_spot_one]
            self.second_bit = parents[1].solution[self.recom_spot_one:self.recom_spot_two]
            self.third_bit = parents[0].solution[self.recom_spot_two:]
        
            new_sol = np.concatenate((self.first_bit, self.second_bit, self.third_bit))
            num_trials += 1
        
            if len(new_sol) == len(set(new_sol)) + 1: # success - no duplicate values except 0
                self.recom = 1
                return new_sol
            if num_trials == max_trials:
                return
            
    def inversion(self, inversion_rate = 0.01):
        """ Gene inversion
        
        One-point inversion of solution. Invert the second part of the solution
        and conatenate it to the first part.
        """ 
        
        if np.random.uniform() <= inversion_rate:
            if len(self.solution) <= 3:
                return
            
            inversion_pt = np.random.choice(list(range(1, len(self.solution) - 1)))
            
            self.first_part = self.solution[:inversion_pt]
            self.second_part = np.flipud(self.solution[inversion_pt:len(self.solution)-1])
            self.second_part = np.concatenate((self.second_part, [0]))
            # keep the solution until the recombination point, invert the remaining bits
            
            self.solution = np.concatenate((self.first_part, self.second_part))
            self.inversions.append([self.first_part, self.second_part])
        
    def mutation(self, mutation_rate = 0.05):
        """ Mutate the solution
        
        Gene by gene, attempt a mutation with probability of mutation_rate.
        If a mutation is chosen, and if there are nodes not within the solution,
        choose either to swap two genes, or choose a gene not in the solution
        and swap out with the current gene
        
        """
        
        for i in range(1, len(self.solution) - 1): # can't mutate first or last bit
             # which sites are not in the solution             
            if (np.random.uniform() <= mutation_rate):
                choices = list(set(list(range(1, Environment.NUM_BIKES + 1))) - set(self.solution))
                
                if (len(choices) > 0) & (np.random.random() <= 0.5):
                    
                    extra_dist = np.zeros(shape = len(choices))

                    for j, b in enumerate(choices):
                        extra_dist[j] = Environment.DIST_M[self.solution[i-1], choices[j]] + \
                                        Environment.DIST_M[choices[j], self.solution[i]] - \
                                        Environment.DIST_M[self.solution[i-1], self.solution[i]]
                                        
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
        """ Add a gene to the solution
        
        If there are nodes not visited by the current solution, attempt to add
        a new gene to the solution with rate new_gene_rate
        """

        if new_gene_rate is None:
            new_gene_rate = 0.08

        choices = list(set(list(range(1, Environment.NUM_BIKES + 1))) - set(self.solution))
        
        if (np.random.uniform() <= new_gene_rate) and (len(choices) > 0):
            new_gene_loc = np.random.randint(1, len(self.solution)) # can't insert into first or last
 
            extra_dist = np.zeros(shape = len(choices))

            for i, b in enumerate(choices):
                extra_dist[i] = Environment.DIST_M[self.solution[new_gene_loc-1], choices[i]] + \
                                Environment.DIST_M[choices[i], self.solution[new_gene_loc]] - \
                                Environment.DIST_M[self.solution[new_gene_loc-1], self.solution[new_gene_loc]]
                              
                                
            # Choose a new gene. Make it very likely that the new gene is close
            # to a gene that already exists. 
            extra_dist_magnified = (max(extra_dist) - extra_dist + 1) ** 10
            extra_dist_final = extra_dist_magnified/extra_dist_magnified.sum()
            
            new_gene = np.random.choice(choices, p = extra_dist_final)
 
            self.insertions.append([(new_gene_loc, new_gene)])
            self.solution = np.insert(self.solution, obj = new_gene_loc, values = new_gene)
            self.solution_size += 1
    
    def get_distance(self, dist_matrix):
        """Return the distance that the solution takes
        
        Returns the distance that the solution takes, taking into account the 
        'lost' distance that stopping to pick up a scooter takes (stop time and
        average speed)
        
        """
        self.dist = 0
        
        for i in range(0, len(self.solution) - 1):
            current_loc = self.solution[i]
            next_loc = self.solution[i+1]
            
            self.dist = self.dist + dist_matrix[current_loc, next_loc] + Environment.STOP_TIME * Environment.AVG_SPEED
        
        return self.dist

    def get_fitness(self):
        return self.solution_size

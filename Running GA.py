# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:09:43 2018

@author: Konrad

"""
import sys
sys.path.insert(0, "C:/Users/Konrad/Desktop/GeneticAlgorithm")
import firstGA as GA

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import mode
from math import log

#bike_id = list(range(num_bikes))

class Environment():

    #global self.algs
       
    def __init__(self, num_bikes = 150, max_dist = 50, pop_size = 30, reproducing_frac = 0.10):
        self.algs = []  
        
        self.num_bikes = num_bikes
        self.max_dist = max_dist
        self.pop_size = pop_size
        self.num_reproducing = int(pop_size * reproducing_frac)
           
        df = pd.DataFrame({'x':np.random.random(size = num_bikes), 'y':np.random.random(size = num_bikes)})
        shop = pd.DataFrame([[0,0]], columns = ['x', 'y'])
        self.df = df.append(shop)
        
        self.dist_m = distance_matrix(self.df, self.df)
        
        test_df = pd.DataFrame([[0,1], [1,1], [1,0], [0,0.5], [0,0]])
        self.test_dist_m = distance_matrix(test_df, test_df)
        
        while len(self.algs) < self.pop_size:
            new_alg = GA.Algorithm(parents = False)

            if new_alg.get_distance(self.dist_m) <= self.max_dist: # feasible solution
                self.algs.append(new_alg)
        
    def find_parents(self):
        max_val = sorted([a.get_fitness() for a in self.algs], reverse = True)[self.num_reproducing-1]
        self.algs = [a for a in self.algs if a.get_fitness() >= max_val] # overwriting the algs to just the fittest
        #return potential_parents
        
        
    
    def next_generation(self):
        pot_par = self.algs[:]
        
        total_fitness = sum([p.get_fitness() for p in pot_par])
        repr_prob = [p.get_fitness()/total_fitness for p in pot_par]
        
        while len(self.algs) < self.pop_size:
            first_par = np.random.choice(pot_par, p = repr_prob)
            second_par = np.random.choice(pot_par, p = repr_prob)
            
            while first_par == second_par:
                second_par = np.random.choice(pot_par, p = repr_prob)
            
            new_alg = GA.Algorithm(parents = [first_par, second_par])
            
            if new_alg.get_distance(self.dist_m) <= self.max_dist: # feasible solution
                self.algs.append(new_alg)
        
    def summarize_generations(self, i):
        num_algs = len(self.algs)
        avg_fitness = round(sum([a.get_fitness() for a in self.algs])/num_algs, 3)
        avg_distance = round(sum(a.get_distance(self.dist_m) for a in self.algs)/num_algs, 3)
        
        best_fitness = max([a.get_fitness() for a in self.algs])
        
        med_fitness = mode([len(a.solution) for a in self.algs], axis = None)[0]
        array_list = [a.solution for a in self.algs if len(a.solution) == med_fitness]
        
        avg_entropy = calculate_entropy(array_list)/med_fitness
        
        print("Iteration: {}\t AvgFitness: {}\t Best Fitness: {}\t AvgDistance: {}\t Avg Entropy:{}"
              .format(i, avg_fitness, best_fitness, avg_distance, avg_entropy))
        
    def run_simulation(self, num_iter):
        
        self.summarize_generations(i = 0)
        
        for i in range(num_iter):
            self.find_parents()
            self.next_generation()
            self.summarize_generations(i+1)

    def get_best(self):
        best_fitness = max([a.get_fitness() for a in self.algs])
        best_sol = [a for a in self.algs if a.get_fitness() == best_fitness]
        
        return best_sol

def calculate_entropy(array_list): # adapted from https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    ent = 0.
    
    if len(array_list) < 2:
        return 0
    
    for i in range(len(array_list)):
        entry = [a[i] for a in array_list]
        value,counts = np.unique(entry, return_counts=True)
        probs = counts / len(entry)
        
        for i in probs:
            ent -= i * log(i, 2)

    return ent

one_trial = Environment(pop_size = 30, max_dist = 10)
one_trial.run_simulation(num_iter = 100)

#test_list = [[0,1], [1,0]]

#calculate_entropy(test_list)

bike_route = one_trial.get_best()[0].solution

one_trial.df.to_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/bike_locations.csv")
np.savetxt("C:/Users/Konrad/Desktop/GeneticAlgorithm/route.csv", bike_route, delimiter = ",")

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:09:43 2018

@author: Konrad

"""
import sys
sys.path.insert(0, "C:/Users/Konrad/Desktop/GeneticAlgorithm")
import firstGA as GA
from collections import Counter
import numpy as np
import pandas as pd
import operator
from scipy.spatial import distance_matrix
from scipy.stats import mode
from math import log

df = pd.read_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/df.csv")
dist_m = pd.read_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/dist.csv")
num_bikes = df.shape[0] - 1
#test_df = pd.DataFrame([[0,0], [0,0.5], [0,1], [1,1], [1,0]])
#test_df.columns = ['x', 'y']
#test_dist_m = distance_matrix(test_df, test_df)

#df = pd.DataFrame({'x':np.random.random(size = num_bikes), 'y':np.random.random(size = num_bikes)})
#shop = pd.DataFrame([[0,0]], columns = ['x', 'y'])
#df = df.append(shop)
#df
#dist_m
#dist_m = distance_matrix(df, df)

class Environment():
       
    def __init__(self, dist_m = dist_m, bikes_df = df, num_bikes = 150, max_dist = 50, 
                 pop_size = 30, reproducing_frac = 0.10):
        self.algs = []  
        self.all_sols = []
        self.gen_best = []
        
        self.dist_m = dist_m
        self.df = bikes_df
        
        self.num_bikes = num_bikes
        self.max_dist = max_dist
        self.pop_size = pop_size
        self.num_reproducing = int(pop_size * reproducing_frac)
        
        while len(self.algs) < self.pop_size:
            new_alg = GA.Algorithm(parents = False, i = 0)

            if new_alg.get_distance(self.dist_m) <= self.max_dist: # feasible solution
                self.algs.append(new_alg)
        
    def find_parents(self):
        self.algs = sorted(self.algs, key = operator.attrgetter('dist'))
        self.algs = sorted(self.algs, key = operator.attrgetter('solution_size'), reverse = True)[:self.num_reproducing-1]

        #self.algs = sorted(self.algs, key = operator.attrgetter('dist'))

        #print([(round(a.solution_size, 3), round(a.dist, 3)) for a in self.algs])
        #self.algs = self.algs
        #min_repr_fitness = sorted([a.get_fitness() for a in self.algs], reverse = True)[self.num_reproducing-1]
        #self.algs = [a for a in self.algs if a.get_fitness() >= min_repr_fitness] # overwriting the algs to just the fittest
            
    def next_generation(self, i):
        pot_par = self.algs[:]
        
        #total_fitness = sum([p.get_fitness() for p in pot_par])
        #repr_prob = [p.get_fitness()/total_fitness for p in pot_par]
        num_par = len(pot_par)
        repr_prob = sorted(list(range(0, num_par)), reverse = True)
        repr_prob_sum = sum(repr_prob)
        repr_prob = [r/repr_prob_sum for r in repr_prob]
        
        while len(self.algs) < self.pop_size:
            first_par = np.random.choice(pot_par, p = repr_prob)
            second_par = np.random.choice(pot_par, p = repr_prob)
            
            while first_par == second_par:
                second_par = np.random.choice(pot_par, p = repr_prob)
            
            new_alg = GA.Algorithm(parents = [first_par, second_par], i = i)
            
            if new_alg.get_distance(self.dist_m) <= self.max_dist: # feasible solution
                self.algs.append(new_alg)
        
    def summarize_generations(self, i):
        num_algs = len(self.algs)
        avg_fitness = round(sum([a.get_fitness() for a in self.algs])/num_algs, 3)
        avg_distance = round(sum(a.get_distance(self.dist_m) for a in self.algs)/num_algs, 3)
        
        best_fitness = max([a.get_fitness() for a in self.algs])
        best_fitness_dist = round(min([a.get_distance(self.dist_m) for a in self.algs if a.get_fitness() == best_fitness]), 3)
        med_fitness = mode([len(a.solution) for a in self.algs], axis = None)[0]
        array_list = [a.solution for a in self.algs if len(a.solution) == med_fitness]
        
        avg_entropy = calculate_entropy(array_list)/(med_fitness - 2) # start and end are fixed  \
        
        print("Iteration: {}\t AvgFitness: {}\t Best Fitness: {}\t BestDistance: {}\t Avg Entropy:{}"
              .format(i, avg_fitness, best_fitness, best_fitness_dist, avg_entropy))
        
    def run_simulation(self, num_iter):
        
        self.summarize_generations(i = 0)
        self.unique_sols = np.zeros(num_iter)       
        for i in range(num_iter):
            current_solutions = [a.solution for a in self.algs]
            self.all_sols.append(current_solutions)
            current_all_sols = [a for b in self.all_sols for a in b]
            self.unique_sols[i] = len(Counter(tuple(a) for a in current_all_sols))
            self.find_parents()
            self.next_generation(i)
            self.summarize_generations(i+1)
            self.gen_best.append(self.get_best())

    def get_best(self):
        best_fitness = max([a.get_fitness() for a in self.algs])
        min_dist = min([a.get_distance(self.dist_m) for a in self.algs if a.get_fitness() == best_fitness])
        best_sol = [a for a in self.algs if (a.get_fitness() == best_fitness) and (a.get_distance(self.dist_m) == min_dist)][0]
        
        return best_sol, min_dist, best_sol.solution

def calculate_entropy(array_list): # adapted from https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    ent = 0.
    
    if len(array_list) < 2:
        return 0
    
    for i in range(len(array_list[0])):
        entry = [a[i] for a in array_list]
        value,counts = np.unique(entry, return_counts=True)
        probs = counts / len(entry)
        
        for i in probs:
            ent -= i * log(i, 2)

    return round(ent, 3)

one_trial = Environment(num_bikes = num_bikes, pop_size = 90, max_dist = 800, reproducing_frac = 0.1,
                        dist_m = dist_m, bikes_df = df)
one_trial.run_simulation(num_iter = 50)
#[(a.get_fitness(), a.dist) for a in one_trial.algs]
one_trial.gen_best
min_dist = min([a[1] for a in one_trial.gen_best])
best_sol = [a[2] for a in one_trial.gen_best if a[1] == min_dist][0]

np.savetxt("C:/Users/Konrad/Desktop/GeneticAlgorithm/route.csv", best_sol, delimiter = ",")
one_trial.df.to_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/bike_locations.csv")

#one_trial.get_best()[0].parents
#one_trial.get_best()[0].solution

#one_trial.get_best()[0].solution_size
#one_trial.get_best()[0].swaps
#one_trial.get_best()[0].mutations
#one_trial.get_best()[0].trial_no

#[list(a.solution) for a in one_trial.algs]
#one_trial.unique_sols.tolist()
#one_trial.all_sols
#one_trial.get_best()[0].recom

#[a.parents for a in one_trial.algs]
#[a.inversions for a in one_trial.algs]
#[a.solution for a in one_trial.algs]
#one_trial.algs[49].parents
#one_trial.algs[49].swaps
#one_trial.algs[49].solution
#one_trial.algs[49].swaps
#one_trial.algs[49].inversions


#best_sol


#one_trial.get_best()[0].get_distance(one_trial.dist_m)
#test_df
#test_dist_m
#one_trial.get_best()[1].solution

#[(a.solution, a.id) for a in one_trial.get_best()]
#[a.inversions for a in one_trial.algs if a.id == 'f8c1c439-e203-11e8-97db-aeb6d0d9cb77']
#import cProfile
#import re
#cProfile.run(one_trial.run_simulation(num_iter = 5))
#cProfile.run('re.compile("foo|bar")')
#cProfile.run('re.compile("one_trial.run_simulation(num_iter = 20)")')

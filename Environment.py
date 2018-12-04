# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:27:56 2018

@author: Konrad
"""
import sys
sys.path.insert(0, "C:/Users/Konrad/Desktop/GeneticAlgorithm")
import GA

import numpy as np
import operator
from scipy.stats import mode
from math import log
import time

num_bikes = GA.num_bikes

class Environment():
       
    def __init__(self, num_bikes, max_dist, 
                 pop_size, reproducing_frac, dist_m = GA.dist_m, bikes_df = GA.df):
        self.algs = []  
        self.all_sols = [] # all solutions, currently unused
        self.gen_best = [] # best in the generation
        self.done_iterations = 0
        self.dist_m = dist_m
        self.df = bikes_df
        
        self.num_bikes = num_bikes
        self.max_dist = max_dist
        self.pop_size = pop_size
        self.num_reproducing = int(pop_size * reproducing_frac)
        
        while len(self.algs) < self.pop_size: #Initializing the population of GAs
            new_alg = GA.Algorithm(parents = False, i = 0)

            if new_alg.get_distance(self.dist_m) <= self.max_dist: # feasible solution
                self.algs.append(new_alg)
        
    def find_parents(self):
        # For rank order selection: GAs are sorted DESC by solution_size with tiebreaker being dist ASC
        self.algs = sorted(self.algs, key = operator.attrgetter('dist')) #sort by dist asc
        self.algs = sorted(self.algs, key = operator.attrgetter('solution_size'), reverse = True)[:self.num_reproducing-1] #sort by solution length
            
    def next_generation(self, i):
        pot_par = self.algs[:]
        self.infeasible = 0 # count infeasible solutions over simulation

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
            else:
                self.infeasible += 1
        
    def summarize_generations(self, i):
        num_algs = len(self.algs)
        avg_fitness = round(sum([a.get_fitness() for a in self.algs])/num_algs, 3)
        
        best_fitness = max([a.get_fitness() for a in self.algs])
        best_fitness_dist = round(min([a.get_distance(self.dist_m) for a in self.algs if a.get_fitness() == best_fitness]), 3)
        mode_fitness = mode([len(a.solution) for a in self.algs], axis = None)[0]
        array_list = [a.solution for a in self.algs if len(a.solution) == mode_fitness]
        self.summary[i] = best_fitness_dist
        
        if mode_fitness:
            avg_entropy = calculate_entropy(array_list)/(mode_fitness - 2) # start and end are fixed  \
        else:
            avg_entropy = np.NaN
        print("Iteration: {}\t AvgFitness: {}\t Best Fitness: {}\t BestDistance: {}\t Avg Entropy:{}"
              .format(i, avg_fitness, best_fitness, best_fitness_dist, avg_entropy))
        
    def run_simulation(self, num_iter, max_time):
        start_time = time.time()
        self.summary = np.zeros(shape = num_iter+1)
        #self.all_algs = []
    
        for i in range(num_iter):
            self.summarize_generations(i)    
            #self.all_algs.append(self.algs)
            self.find_parents()
            self.next_generation(i)
            self.gen_best.append(self.get_best())
            self.done_iterations += 1
            
            if time.time() - start_time >= max_time:
                return

    def get_best(self):
        best_fitness = max([a.get_fitness() for a in self.algs])
        min_dist = min([a.get_distance(self.dist_m) for a in self.algs if a.get_fitness() == best_fitness])
        best_sol = [a for a in self.algs if (a.get_fitness() == best_fitness) and (a.get_distance(self.dist_m) == min_dist)][0]
        
        return best_sol, best_fitness, min_dist, best_sol.solution

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
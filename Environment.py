# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:27:56 2018

@author: Konrad
"""
import sys
sys.path.insert(0, "C:/Users/Konrad/Desktop/GeneticAlgorithm")
import GA
#from collections import Counter
#import pandas as pd
#from scipy.spatial import distance_matrix
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
        self.all_sols = []
        self.gen_best = []
        self.done_iterations = 0
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
            
    def next_generation(self, i):
        pot_par = self.algs[:]
        self.infeasible = 0
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
            else:
                self.infeasible += 1
        
    def summarize_generations(self, i):
        num_algs = len(self.algs)
        avg_fitness = round(sum([a.get_fitness() for a in self.algs])/num_algs, 3)
        #avg_distance = round(sum(a.get_distance(self.dist_m) for a in self.algs)/num_algs, 3)
        
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
        self.all_algs = []
        #self.unique_sols = np.zeros(num_iter)       
        for i in range(num_iter):
            self.summarize_generations(i)    
            self.all_algs.append(self.algs)
            #current_solutions = [a.solution for a in self.algs]
            #self.all_sols.append(current_solutions)
            #current_all_sols = [a for b in self.all_sols for a in b]
            #self.unique_sols[i] = len(Counter(tuple(a) for a in current_all_sols))
            self.find_parents()
            self.next_generation(i)
 #           self.summarize_generations(i)
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
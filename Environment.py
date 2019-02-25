# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:27:56 2018

@author: Konrad
"""
import os
os.chdir("C:/Users/Konrad/Desktop/GeneticAlgorithm")

import sys
sys.path.insert(0, "C:/Users/Konrad/Desktop/GeneticAlgorithm")
import GA

import numpy as np
import operator
from scipy.stats import mode
import time
import pandas as pd
import helpers

DF = pd.read_csv("Data/df.csv")
DIST_M = np.loadtxt(open("Data/dist.csv", "rb"), delimiter=",", skiprows=1)

NUM_BIKES = DF.shape[0] - 1

bike_dist = np.loadtxt(open("Data/workshopdist.csv", "rb"), delimiter=",", skiprows=1)
bike_dist = (max(bike_dist) - bike_dist + 1) ** 10 # seed solutions to start nearby the warehouse first
BIKE_PROB = bike_dist/bike_dist.sum()

BIKE_PROB /= BIKE_PROB.sum()

STOP_TIME = 10 # minutes picking up a scooter
AVG_SPEED = 400 # meters/minute average speed


class Environment():
    """ Creates an environment for the GAs to find solutions. 
    
    Takes in a maximum distance for the solution, the number of GAs, and
    how many of the GAs reproduce. 
    
    """
    
    def __init__(self, max_dist, 
                 pop_size, reproducing_frac):
        self.algs = []  
        self.all_sols = [] # all solutions, currently unused
        self.gen_best = [] # best in the generation
        self.done_iterations = 0
        self.dist_m = DIST_M
        self.df = DF
        
        self.num_bikes = NUM_BIKES
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
            avg_entropy = helpers.calculate_entropy(array_list)/(mode_fitness - 2) # start and end are fixed  \
        else:
            avg_entropy = np.NaN
        print("Iteration: {}\t AvgFitness: {}\t Best Fitness: {}\t BestDistance: {}\t Avg Entropy:{}"
              .format(i, avg_fitness, best_fitness, best_fitness_dist, avg_entropy))
        
    def run_simulation(self, num_iter, max_time):
        start_time = time.time()
        self.summary = np.zeros(shape = num_iter+1)
    
        for i in range(num_iter):
            self.summarize_generations(i)    
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


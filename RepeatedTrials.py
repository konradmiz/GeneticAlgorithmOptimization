# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:18:37 2018

@author: Konrad
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:29:06 2018

@author: Konrad
"""
#import sys
import os

import numpy as np
import pandas as pd

import argparse
#sys.path.insert(0, "C:/Users/Konrad/Desktop/GeneticAlgorithm")
os.chdir("C:/Users/Konrad/Desktop/GeneticAlgorithm")
import GA
import Environment as ENV

parser = argparse.ArgumentParser(description='Input max distance and number of trials')

parser.add_argument('-max_dist', metavar='max_dist', type=int,
                    help='Maximum allowable distance', nargs='?')
parser.add_argument('-num_iter', metavar = 'number_of_iterations', type = int, help = 'number of trials to run', 
                    nargs = '?')
parser.add_argument('-pop_size', metavar = 'population_size', type = int, help = 'population size of trial', 
                    nargs = '?')
parser.add_argument('-max_time', metavar = 'maximum_time', type = int, help = 'upper bound of trial run duration', 
                    nargs = '?')
parser.add_argument('-repr_frac', metavar = 'reproducing_fraction', type = float, help = 'fraction of population reproducing', 
                    nargs = '?')

args = parser.parse_args()

max_dist = args.max_dist if args.max_dist else 50000
num_iter = args.num_iter if args.num_iter else 100
pop_size = args.pop_size if args.pop_size else 100
max_time = args.max_time if args.max_time else 180
reproducing_frac = args.repr_frac if args.repr_frac else 0.15

fitness_distance_summary = pd.DataFrame({'fitness':[], 'distance':[], 'trial':[]})
for i in range(10):
    one_trial = ENV.Environment(num_bikes = GA.num_bikes, pop_size = pop_size,
                                max_dist = max_dist, reproducing_frac = reproducing_frac)
    
    one_trial.run_simulation(num_iter = num_iter, max_time = max_time)

    trial_summary = pd.DataFrame({'fitness':[a[1] for a in one_trial.gen_best], 
                                  'distance':[a[2] for a in one_trial.gen_best],
                                  'trial':[i] * num_iter})
    fitness_distance_summary = fitness_distance_summary.append(trial_summary)

fitness_distance_summary.to_csv('Results/summary.csv')


one_trial.all_sols

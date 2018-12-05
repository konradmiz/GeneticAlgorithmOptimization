# GeneticAlgorithmOptimization

Building a flexible GA from scratch in Python to solve an optimization problem relating to bike maintenance. 

## Workflow

All relevant code is in the top-level directory. 

RunningGA.ps1 is the PowerShell script that runs scripts for 

* ingesting data (IngestingData.R)

* running the GA simulation (RunningSimulation.py)

* visualizing the GA results (VisualizingResults.R)

Two additional Python modules exist, that define the GA (GA.py) and the environment/population of GAs (Environment.py). 

These files all assume a directory called GeneticAlgorithm, with Data, Results, and Images subfolders. Since this was a solo project the paths are hard-coded in. 

## Additional code

The RepeatedTrials.py script is not currently used; it runs the simulation multiple times and could be used to e.g. compare the solutions between trials. 

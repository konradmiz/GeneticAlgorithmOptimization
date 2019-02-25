# GeneticAlgorithmOptimization

Building a flexible GA from scratch in Python to solve an optimization problem relating to bike maintenance. 

**Of note: this code will not run, as I have not included database credentials to this repo. Proxy data may be uploaded sometime in the future. Additionally, the filepaths are hard-coded in.**  

## Workflow

All relevant code is in the top-level directory. 

running_ga.ps1 is the PowerShell script that runs the simulation, i.e. these scripts:

* ingesting data (IngestingData.R)

* running the GA simulation (RunningSimulation.py)

* visualizing the GA results (VisualizingResults.R)

Two additional Python modules exist, that define the GA (GA.py) and the environment/population of GAs (Environment.py). 

These files all assume a directory called GeneticAlgorithm, with Data, Results, and Images subfolders. 

## Additional code

The RepeatedTrials.py script is not currently used; it runs the simulation multiple times and could be used to e.g. compare the solutions between trials. 

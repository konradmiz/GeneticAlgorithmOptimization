# PowerShell script to run the GeneticAlgorithm code. 
# The script is called with parameters which are then passed
# to the data preparation script, the script that runs the GA simulation, 
# and the script that visualizes the results. 

# The script is called akin to this when PowerShell is initialized and is in the correct directory:
# ./running_ga.ps1 -vehicle "scooters" -max_dist 100000 -num_iter 10

param([string]$vehicle="bikes", 
      [Int32]$max_dist, 
      [Int32]$num_iter, 
      [Int32]$pop_size, 
      [Int32]$max_time,
      [Float]$repr_frac)
        
cd C:\Users\Konrad\Desktop\GeneticAlgorithm

RScript IngestingData.R -v $vehicle # creates tables
python.exe RunningSimulation.py -max_dist $max_dist -num_iter $num_iter -pop_size $pop_size -max_time $max_time -repr_frac $repr_frac
RScript VisualizingResults.R
start "Images\Routes.png"
start "Images\FitnessOverTime.png"

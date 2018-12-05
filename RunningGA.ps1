param([string]$vehicle="bikes", 
      [Int32]$max_dist, 
      [Int32]$num_iter, 
      [Int32]$pop_size, 
      [Int32]$max_time,
      [Float]$repr_frac)
        
cd C:\Users\Konrad\Desktop\GeneticAlgorithm

RScript IngestingData.R -v $vehicle
python.exe RunningSimulation.py -max_dist $max_dist -num_iter $num_iter -pop_size $pop_size -max_time $max_time -repr_frac $repr_frac
RScript VisualizingResults.R
start "Images\Routes.png"
start "Images\FitnessOverTime.png"

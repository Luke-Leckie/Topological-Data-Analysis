#!/bin/bash
#SBATCH -A r01531
#SBATCH -J leckie_TDA
#SBATCH --mail-user=lleckie@iu.edu
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=500000M
#SBATCH --array=0-11

# Path to your parameter file
PARAM_FILE=/N/u/lleckie/Quartz/work/TDA_cluster/parameters.txt

# Get the line corresponding to this array task.
# Note: SLURM_ARRAY_TASK_ID is zero-indexed so we add 1 for sed.
PARAM=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $PARAM_FILE)
overlap=$(echo "$PARAM" | cut -f1)
window=$(echo "$PARAM" | cut -f2)
df_name=$(echo "$PARAM" | cut -f3)

echo "Running with parameters:"
echo "Overlap: $overlap"
echo "Window:  $window"
echo "DF Name: $df_name"

# Get full python path (if needed)
full_python_path=$(which python3)
echo "Python Path: $full_python_path"

# Call the Python script and pass the parameters as arguments.
python3 /N/u/lleckie/Quartz/work/TDA_cluster/TDA_pt1_process_narratives_cluster.py "$overlap" "$window" "$df_name"


#!/bin/bash

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=16                  # Run a single task
#SBATCH --partition=nova #priority-a100    # gpu node(s)
#SBATCH -A mech-ai
#SBATCH --mem=50G   # maximum memory per node
#SBATCH --job-name=triplane_datagen   # Job name
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=8:00:00 #Time limit hrs:min:sec
#SBATCH --array=0-19

source ~/.bashrc_old

module load miniconda3
module load parallel
source activate nfd

echo "beginning executing cases..."

ID=0 # 1,2,3,4 <-- OBJ IDs

parallel="parallel --controlmaster --delay .2 -j $SLURM_NTASKS --joblog logs/runtask_${ID}_${SLURM_ARRAY_TASK_ID}.log"

$parallel "python /work/mech-ai/jrrade/Tri-plane/NFD/nfd/triplane_decoder/generate_3d_dataset.py --input {1}/models/model_normalized.obj --output {1}/models/model_normalized.npy --type border --count 1000000" :::: args/args_${ID}_${SLURM_ARRAY_TASK_ID}.txt
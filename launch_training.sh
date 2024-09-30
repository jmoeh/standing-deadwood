#!/bin/bash
#SBATCH --ntasks=8
#SBATCH -J deadwood-ortho
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# load python version and virtual environment
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /home/sc.uni-leipzig.de/jk947skaa/standing-deadwood/venv/bin/activate

export NCCL_P2P_DISABLE=1

WORKSPACE="/lscratch/standing-deadwood"
EXPERIMENT="20k_100epochs_vanilla_pos_weight_12_bce_09"

# create workspace dir if not exists
if [ ! -d "$WORKSPACE" ]; then
    echo "$WORKSPACE does not exist"
    mkdir -p $WORKSPACE
else
    echo "$WORKSPACE already exists"
    # rm -rf $WORKSPACE/*
fi

# echo "copy tiles"
# echo $(date)
# rsync -ah --progress ~/work/tiles_1024.tar $WORKSPACE
# echo "copy done..."

# echo $(date)
# echo "untar tiles"
# tar -xf $WORKSPACE/tiles_1024.tar -C $WORKSPACE
# echo "untar done..."

LAUNCHER="accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_processes=8
    --num_machines=1
    --rdzv_conf rdzv_backend=static \
    /home/sc.uni-leipzig.de/jk947skaa/standing-deadwood/train.py \
    --fold $SLURM_ARRAY_TASK_ID \
    --config /home/sc.uni-leipzig.de/jk947skaa/experiments/$EXPERIMENT/config.json

"

$LAUNCHER

rsync -ah $WORKSPACE/$EXPERIMENT/* /home/sc.uni-leipzig.de/jk947skaa/experiments/$EXPERIMENT

echo $(date)
echo "removing data"
rm -rf $WORKSPACE

echo "remove done"

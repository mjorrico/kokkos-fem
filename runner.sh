#!/bin/bash -l

#SBATCH -A uppmax2024-2-16
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 0:20:00
#SBATCH --gres=gpu:1 --gpus-per-node=1
#SBATCH -J ABP_task_4
#SBATCH -D ./
#SBATCH --output=temp.out

module load gcc/12.2 cuda/12.2.2

./fet_double_left.cuda -min 1e4 -max 1e7 -repeat 20 > output/fet_double_left_cuda.txt
./fet_double_right.cuda -min 1e4 -max 1e7 -repeat 20 > output/fet_double_right_cuda.txt
./fet_float_left.cuda -min 1e4 -max 1e7 -repeat 20 > output/fet_float_left_cuda.txt
./fet_float_right.cuda -min 1e4 -max 1e7 -repeat 20 > output/fet_float_right_cuda.txt

./fet_double_left.host -min 1e4 -max 1e7 -repeat 20 > output/fet_double_left_host.txt
./fet_double_right.host -min 1e4 -max 1e7 -repeat 20 > output/fet_double_right_host.txt
./fet_float_left.host -min 1e4 -max 1e7 -repeat 20 > output/fet_float_left_host.txt
./fet_float_right.host -min 1e4 -max 1e7 -repeat 20 > output/fet_float_right_host.txt
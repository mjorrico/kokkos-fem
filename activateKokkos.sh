#!/usr/bin/bash

# Load GCC module
module load gcc/12.2.0

# Add module paths
module use /sw/EasyBuild/snowy/modules/all/
module use /sw/EasyBuild/snowy-gpu/modules/all/

# Load CUDA module
module load CUDA/12.2.0


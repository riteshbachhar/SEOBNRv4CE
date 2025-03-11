#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH --mem=60G
#SBATCH -c 64
#SBATCH -p cpu
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-user=user@domain.edu
#SBATCH --mail-type=ALL

source path-to/seobnrv4ce-venv/bin/activate

# For the SEOBNRv4_ROM model implemented in LALSuite to work
# LAL_DATA_PATH needs to point to a directory which contains SEOBNRv4ROM_v2.0.hdf5 or SEOBNRv4ROM_v3.0.hdf5,
# depending on your LALSuite version. Please see the README file.
export LAL_DATA_PATH=PATH-TO-DATAFILE/
export OMP_NUM_THREADS=1

bilby_analyze_injection \
  --signal_approximant NRHybSur3dq8_22 \
  --template_approximant SEOBNRv4CE \
  --distance 205.0 \
  --mass-ratio 0.25 \
  --chi_1 0.5 \
  --chi_2 0.5 \
  --phase_marginalization \
  --distance_marginalization \
  --npool 64 \
  --naccept 120 \
  --npoints 1024 \
  --maxmcmc 5000 \
  --zero-noise

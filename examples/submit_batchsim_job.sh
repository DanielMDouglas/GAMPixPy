#!/bin/bash

#SBATCH --partition=turing
#SBATCH --account=mli:gampix
#
#SBATCH --job-name=gmpixpy
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH --gpus=1
#
#SBATCH --time=5:00:00

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub22.04-cuda12.1-pytorch2.2.1-larndsim.sif

INPUT_EDEPSIM=$1
OUTPUT_HDF5=$2

TMPID=$(cat /proc/sys/kernel/random/uuid) 

GAMPIXROOT=${HOME}/studies/GAMpix/GAMPixPy

COMMAND="python3 ${GAMPIXROOT}/examples/batch_sim.py ${INPUT_EDEPSIM} -o tmp/${TMPID}.h5 -r ${GAMPIXROOT}/gampixpy/readout_config/GAMPixD.yaml -d ${GAMPIXROOT}/gampixpy/detector_config/coh_250.yaml"

singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

COMMAND="python3 ${GAMPIXROOT}/examples/rotate_to_experimental_coordinates.py tmp/${TMPID}.h5 -o ${OUTPUT_HDF5} -d ${GAMPIXROOT}/gampixpy/detector_config/coh_250.yaml"

singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}


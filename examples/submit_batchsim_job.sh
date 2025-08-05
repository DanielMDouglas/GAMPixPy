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

# INPUT_EDEPSIM=$1
# GAMPIXPY_OUTPUT=$2
# ROTATED_OUTPUT=$3

INPUT_EDEPSIM=$1
ROTATED_OUTPUT=$2

ID=$(cat /proc/sys/kernel/random/uuid) 
TMP_GAMPIXPY_OUTPUT=/lscratch/dougl215/tmp/${ID}.h5

# READOUT_CONFIG=$3
# DETECTOR_CONFIG=$4

GAMPIXROOT=${HOME}/studies/GAMpix/GAMPixPy

COMMAND="python3 ${GAMPIXROOT}/examples/batch_sim.py ${INPUT_EDEPSIM} -o ${TMP_GAMPIXPY_OUTPUT} -r ${GAMPIXROOT}/gampixpy/readout_config/GAMPixD.yaml -d ${GAMPIXROOT}/gampixpy/detector_config/coh_250.yaml"
# COMMAND="python3 ${GAMPIXROOT}/examples/batch_sim.py ${INPUT_EDEPSIM} -o ${TMP_FILE} -r ${READOUT_CONFIG} -d ${DETECTOR_CONFIG}"

echo $COMMAND
singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

COMMAND="python3 ${GAMPIXROOT}/examples/rotate_to_experimental_coordinates.py ${TMP_GAMPIXPY_OUTPUT} -o ${ROTATED_OUTPUT} -d ${GAMPIXROOT}/gampixpy/detector_config/coh_250.yaml"
# COMMAND="python3 ${GAMPIXROOT}/examples/rotate_to_experimental_coordinates.py ${TMP_FILE} -o ${OUTPUT_HDF5} -d ${DETECTOR_CONFIG}"

echo $COMMAND
singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

rm $TMP_GAMPIXPY_OUTPUT

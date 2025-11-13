#!/usr/bin/env bash
# run_dataset.sh
# $1 is the HTCondor Process ID (0…9)

JOB_ID="$1"

nvidia-smi
python dataset_generation.py @galaxy_rot_curve_0_1_DAG.txt --sample_seed ${JOB_ID} --job_id ${JOB_ID}

echo "Finished job ${JOB_ID} on `whoami`@`hostname`"

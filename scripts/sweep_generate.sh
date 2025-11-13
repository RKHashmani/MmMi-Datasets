#!/usr/bin/env bash


IDX=$1

nvidia-smi
python sweep_generate.py $IDX

echo "Finished job ${JOB_ID} on `whoami`@`hostname`"
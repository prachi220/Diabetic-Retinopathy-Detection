#!/bin/sh
### Set the job name
#PBS -N disease_grading_model1_10_2_2
### Set the project name, your department dc by default
#PBS -P cse
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M cs5140289@cse.iitd.ac.in
####
#PBS -l select=2:ncpus=4:ngpus=2:mem=50GB

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=100:00:00
#PBS -o stdout_model1_10_2_2
#PBS -e stderr_model1_10_2_2

#### Get environment variables from submitting shell
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
cd $PBS_O_WORKDIR
module load pythonpackages/2.7.13/ucs4/gnu/447/scikit-learn/0.18.1/gnu
module load apps/pythonpackages/2.7.13/tensorflow/1.1.0/gpu
module load apps/pythonpackages/2.7.13/keras/2.1.1/gpu
module load pythonpackages/2.7.13/ucs4/gnu/447/scipy/0.19.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/six/1.10.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/pillow/4.1.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/pandas/0.20.0rc1/gnu

python model1_10_2_2.py
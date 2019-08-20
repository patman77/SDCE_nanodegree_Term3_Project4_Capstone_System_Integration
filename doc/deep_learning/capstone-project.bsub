#!/bin/bash -l
# Sample script for capstone project job

## Scheduler parameters ##

#BSUB -J capstone-project           # job name
#BSUB -o capstone-project.%J.stdout # optional: have output written to specific file
#BSUB -e capstone-project.%J.stderr # optional: have errors written to specific file
# #BSUB -q rb_highend               # optional: use highend nodes w/ Volta GPUs (default: Geforce GPUs)
#BSUB -W 2:00                       # fill in desired wallclock time [hours,]minutes (hours are optional)
#BSUB -n 1                          # min CPU cores,max CPU cores (max cores is optional)
#BSUB -M 2048                       # fill in required amount of RAM (in Mbyte)
# #BSUB -R "span[hosts=1]"          # optional: run on single host (if using more than 1 CPU cores)
# #BSUB -R "span[ptile=28]"         # optional: fill in to specify cores per node (max 28)
# #BSUB -P myProject                # optional: fill in cluster project
#BSUB -R "rusage[ngpus_excl_p=1]"   # use 1 GPU (in explusive process mode)

## Job parameters ##

# Anaconda virtualenv to be used
# Create before running the job with e.g.
# conda create -n tensorflow-3.5 python=3.5 tensorflow-gpu
vEnv=carnd-term3_capstone_project # (please change)

# Source environment (optional)
#. /fs/applications/lsf/latest/conf/profile.lsf
#. /fs/applications/modules/current/init/bash

# Load modules
module purge
module load conda/4.4.8-writecache cudnn/8.0_v6.0 cuda/8.0.0

# Activate environment
source activate $vEnv

# Run your code here (please change, this is only an example)
cat << EOT > carnd-term3_capstone_project.py
# Tensorflow code example on GPU
# from https://www.tensorflow.org/programmers_guide/using_gpu
import tensorflow as tf
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
EOT
python carnd-term3_capstone_project.py

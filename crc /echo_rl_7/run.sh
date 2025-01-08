#!/bin/bash

#$ -M bsun4@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 3        # Specify parallel environment and legal core size
#$ -q gpu            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N echo_project       # Specify job name

source /afs/crc.nd.edu/user/b/bsun4/Private/echo_rl/bin/activate

python3 learning.py 


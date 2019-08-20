GPU cluster

GPUs per node:
- 10x Nvidia GeForce GTX 1080 Ti or 
- 8x Nvidia Tesla V100 (restricted usage)

jobs can be scheduled by calling e.g. bsub < capstone-project.bsub.

All the given parameters (wallclock time, number of CPU cores, number of GPU cores, RAM etc.) at the top of the bsub file must be chosen.
So GPU parallelization is possible, but it must be prepared on tensorflow / keras level.

Alternatively, an interactive job can be started, but it is restricted to 8 hours.

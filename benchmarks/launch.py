import os, sys

for i in [ 1, 2, 4, 8, 16, 32 ]:
    f = open( "launch_{}.sh".format( i ), "w" )
    f.write( """
#!/bin/bash

#PBS -N sdot 
#PBS -P nemesis
#PBS -o output_{}.txt
#PBS -e error.txt
#PBS -l walltime=0:10:00
#PBS -l nodes={}:ppn=4

module load sgi-mpt/2.17
module load gcc/6.4.0

cd $PBS_O_WORKDIR

echo pouetoxe
mpiexec_mpt -n {} build/bench_volume_2D.exe --distribution random --max-dirac-per-cell 10 --nb-diracs=1000000
    """.format( i, i, i ) )
    f.close()

    os.system( "qsub launch_{}.sh".format( i ) )
import matplotlib.pyplot as plt
import os, subprocess
import numpy as np

timings = []
for nmpi in [ 1, 2, 4, 8 ]:
    cmd = "mpirun -np {} build/bench_volume_2D.exe --distribution=random --max-dirac-per-cell=10 --nb-diracs=1000000".format( nmpi )
    res = subprocess.run( cmd.split( " " ), stdout = subprocess.PIPE )
    for l in res.stdout.decode( "utf8" ).split( "\n" ):
        if "Time::delta( t0, t2 )" in l:
            p = float( l.split( " " )[ -1 ] )
            timings.append( p )

print( timings )

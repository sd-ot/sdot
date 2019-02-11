import matplotlib.pyplot as plt
import os, subprocess
import numpy as np

def cmd( method, distrib, nb_diracs ):
    if distrib.endswith( "_w" ):
        distrib = "vtk/{}_{}.xyw".format( distrib[ :-2 ], int( nb_diracs ) )
  
    if method == "recursive":
        return "nsmake run benchmarks/bench_volume_2D.cpp --distribution={} --max-dirac-per-cell=10 --nb-diracs={}".format( distrib, int( nb_diracs ) )
    if method == "neighbor":
        return "nsmake run -DUSE_ZGRID benchmarks/bench_volume_2D.cpp --distribution={} --max-dirac-per-cell=20 --nb-diracs={}".format( distrib, int( nb_diracs ) )
    if method == "cgal":
        return "nsmake run benchmarks/bench_volume_2D_CGAL.cpp --distribution={} --nb-diracs={}".format( distrib, int( nb_diracs ) )

methods = [ "recursive", "neighbor", "cgal" ]
# sizes = [ 1 * 1e5, 2 * 1e5, 4 * 1e5, 8 * 1e5, 16 * 1e5, 32 * 1e5 ]

# for distrib in [ "split_w", "split", "regular", "random_w", "random" ]:
for distrib in [ "random" ]:
    sizes = [ 1e4, 1e5, 1e6, 1e7, 1e8 ]
    if distrib == "split_w":
        sizes = [ 1e4, 1e5 ]

    timings = {}
    for method in methods:
        timings[ method ] = []

    for nb_diracs in sizes:
        for method in methods:
            best = 1e40
            for i in range( 1 + 3 * ( nb_diracs < 1e6 ) ):
                res = subprocess.run( cmd( method, distrib, nb_diracs ).split( " " ), stdout = subprocess.PIPE )
                for l in res.stdout.decode( "utf8" ).split( "\n" ):
                    if "Time::delta( t0, t2 )" in l:
                        p = float( l.split( " " )[ -1 ] )
                        best = min( best, p )
            if best < 1e40:
                timings[ method ].append( best )
    
    print( timings )
    plt.clf()
    plt.loglog( sizes, timings[ "recursive" ] )
    plt.loglog( sizes, timings[ "neighbor" ] )
    plt.loglog( sizes, timings[ "cgal" ] )
    plt.legend( [ "recursive", "neighbor", "cgal" ] )
    plt.xlabel( "nb diracs" )
    plt.ylabel( "time (s)" )
    plt.grid((True,True))
    plt.title( distrib )
    # plt.show()
    plt.savefig( distrib + ".pdf" )
    plt.savefig( distrib + ".png" )

    print( distrib )
    print( "speedup recursive:", np.mean( np.array( timings[ "cgal" ] ) / np.array( timings[ "recursive" ] ) ) )
    print( "speedup neighbor:", np.mean( np.array( timings[ "cgal" ] ) / np.array( timings[ "neighbor" ] ) ) )

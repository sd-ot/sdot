import os, subprocess

def cmd( method, distrib, nb_diracs ):
    if method == "recursive":
        return "nsmake run benchmarks/bench_volume_2D.cpp --distribution={} --max-dirac-per-cell=10 --nb-diracs={}".format( distrib, nb_diracs )
    if method == "neighbor":
        return "nsmake run -DUSE_ZGRID benchmarks/bench_volume_2D.cpp --distribution={} --max-dirac-per-cell=20 --nb-diracs={}".format( distrib, nb_diracs )
    if method == "cgal":
        return "nsmake run benchmarks/bench_volume_2D_CGAL.cpp --distribution={} --nb-diracs={}".format( distrib, nb_diracs )

methods = [ "recursive", "neighbor", "cgal" ]

for distrib in [ "random" ]:
    timings = {}
    for method in methods:
        timings[ method ] = []

    for nb_diracs in [ 1e5, 1e6, 1e7 ]:
        for method in methods:
            res = subprocess.run( cmd( method, distrib, nb_diracs ).split( " " ), stdout = subprocess.PIPE )
            for l in res.stdout.decode( "utf8" ).split( "\n" ):
                if "Time::delta( t0, t2 )" in l:
                    p = float( l.split( " " )[ -1 ] )
                    timings[ method ].append( p )
                    print( method, nb_diracs, p )
    
    print( timings )

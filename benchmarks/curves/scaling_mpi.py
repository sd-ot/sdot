import matplotlib.pyplot as plt

sizes   = [ 1, 2, 4, 16 ]
timings = [ 46.4443, 28.8755, 17.2504, 6.34 ]

plt.loglog( sizes, timings )
# plt.legend( [ "recursive", "neighbor", "cgal" ] )
plt.xlabel( "nb mpi instances" )
plt.ylabel( "time (s)" )
plt.grid((True,True))
plt.ylim( top = 101 ) 
# plt.title( distrib )
plt.savefig( "scaling_mpi.pdf" )
plt.savefig( "scaling_mpi.png" )
plt.show()

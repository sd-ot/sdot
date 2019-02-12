import matplotlib.pyplot as plt

sizes   = [ 1, 2, 4 ]
timings = [ 12.9232, 8.32606, 4.91775 ]

plt.loglog( sizes, timings )
# plt.legend( [ "recursive", "neighbor", "cgal" ] )
plt.xlabel( "nb mpi instances" )
plt.ylabel( "time (s)" )
plt.grid((True,True))
# plt.title( distrib )
plt.show()
plt.savefig( "scaling_mpi.pdf" )
plt.savefig( "scaling_mpi.png" )

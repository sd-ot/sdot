from pysdot import OptimalTransport
from matplotlib import pyplot
import numpy as np

positions = []
ss = 1e-3
for x in np.linspace( 0, 1 - ss, 20 ):
    positions.append( [ x, 0.5 ] )
    positions.append( [ x + ss, 0.5 ] )

ot = OptimalTransport( np.array( positions ) )
ot.verbosity = 1
ot.adjust_weights()

pyplot.plot( ot.get_positions()[ :, 0 ], ot.get_weights(), '+' )
pyplot.show()

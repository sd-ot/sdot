import sys

# type name
out = sys.argv[ 2 ]
tna = out[ out.rindex( "Vec" ) + 3 : out.rindex( "." ) ]

# input data
inp = open( sys.argv[ 1 ] ).read()

# 
open( out, "w" ).write( inp.replace( "TT", tna ) )

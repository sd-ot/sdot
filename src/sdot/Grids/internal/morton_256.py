import sys

sys.stdout.write( "#include <cstdint>\n" );
sys.stdout.write( "\n" );

for s  in [ 32, 64 ]:
    for d in range( 2, 4 ):
        for nd in range( d ):
            sys.stdout.write( "const std::uint{2}_t morton_256_{1}D_{0}_{2}[ 256 ] = ".format( "xyz"[nd], d, str( s ) ) + "{" );
            for i in range( 256 ):
                x = 0
                a = 1
                for b in range( 8 ):
                    if i & pow( 2, b ):
                        x += pow( 2, d * b + nd )
                if i % 16 == 0:
                    sys.stdout.write( "\n    " );
                else:
                    sys.stdout.write( " " );
                sys.stdout.write( "{0:7},".format( x ) );
            sys.stdout.write( "\n};\n" );

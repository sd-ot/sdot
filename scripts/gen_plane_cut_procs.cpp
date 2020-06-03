#include "../src/sdot/support/ERROR.h"
#include "../src/sdot/support/P.h"
#include "internal/GenPlaneCutProc.h"

int main( int argc, char **argv ) {
    // args
    bool random = false;
    bool start_inc = false;
    std::string op_filename, out_filename, scalar_type, size_type, arch;
    for( int n = 1; n < argc; ++n ) {
        std::string arg = argv[ n ];
        if ( arg == "--random" ) {
            random = 1;
            continue;
        }

        if ( arg == "--inc" ) {
            start_inc = true;
            continue;
        }

        //  FP64 ---type U64 --arch
        if ( arg == "--scalar-type" ) {
            scalar_type = argv[ ++n ];
            continue;
        }

        if ( arg == "--size-type" ) {
            size_type = argv[ ++n ];
            continue;
        }

        if ( arg == "--arch" ) {
            arch = argv[ ++n ];
            continue;
        }

        if ( arg == "--op" ) {
            op_filename = argv[ ++n ];
            continue;
        }

        if ( arg == "--out" ) {
            out_filename = argv[ ++n ];
            continue;
        }

        ERROR( "unknown arg type ('{}')", arg );
    }

    //
    OptParm op( op_filename, random );
    if ( start_inc )
        op.inc( false );

    GenPlaneCutProc gpcp( op, scalar_type, size_type, arch );
    std::ofstream fout( out_filename );
    gpcp.gen( fout );

    op.save( op_filename );
}

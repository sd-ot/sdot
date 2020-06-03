#include "../../src/sdot/support/ASSERT.h"
#include "../../src/sdot/support/TODO.h"
#include "../../src/sdot/support/P.h"
#include "PlaneCutProcPlan.h"
#include "GenPlaneCutProc.h"
#include <map>

GenPlaneCutProc::GenPlaneCutProc( OptParm &op, std::string scalar_type, std::string size_type, std::string arch ) : scalar_type( scalar_type ), size_type( size_type ),  arch( arch ), op( op ) {
    size_for_test_during_load = 4;
    size = 8; // + op.get_value( 2 );
}

void GenPlaneCutProc::gen( std::ostream &os ) {
    // stuff for benchmarks + IDEs
    gen_header( os );

    // decl of reg + load data from memory
    os << "\n";
    os << "    if ( nodes_size <= " << size << " ) {\n";
    gen_load( os );

    // beg for loop
    os << "\n";
    os << "        for( ; ; ++num_cut ) {\n";
    os << "            if ( num_cut == nb_cuts ) {\n";
    gen_store( os, 16 );
    os << "                return;\n";
    os << "            }\n";

    // scalar product and case number
    os << "\n";
    gen_sp_and_case( os );

    //
    gen_cases( os );

    // end for loop
    os << "        }\n";

    // stuff for benchmarks + IDEs
    os << "    }\n";
    gen_footer( os );
}

void GenPlaneCutProc::gen_sp_and_case( std::ostream &os ) {
    std::string sp = "            ";
    os << sp << "// get distance and outside bit for each node\n";
    os << sp << "int nmsk = 1 << nodes_size;\n";
    os << sp << "TF nx = nds[ 0 ][ num_cut ];\n";
    os << sp << "TF ny = nds[ 1 ][ num_cut ];\n";
    os << sp << "TF ns = nss[ num_cut ];\n";
    os << sp << "\n";
    for_each_reg( [&]( int off, int len ) {
        os << sp << fv( len ) << " bi" << off << " = px" << off << " * nx + py" << off << " * ny;\n";
    } );

    os << "\n";
    os << sp << "int outside_nodes = ";
    for_each_reg( [&]( int off, int /*len*/ ) {
        os << ( off ? " | " : "" ) << "( ( bi" << off << " > ns ) << " << off << " )";
    } );
    os << ";\n";

    os << sp << "int case_code = ( outside_nodes & ( nmsk - 1 ) ) | nmsk;\n";
    for_each_reg( [&]( int off, int len ) {
        os << sp << fv( len ) << " di" << off << " = bi" << off << " - ns;\n";
    } );
}

void GenPlaneCutProc::gen_load( std::ostream &os ) {
    // reg decl
    for_each_reg( [&]( int off, int len ) { os << "        " << fv( len ) << " px" << off << ";\n"; } );
    for_each_reg( [&]( int off, int len ) { os << "        " << fv( len ) << " py" << off << ";\n"; } );
    for_each_reg( [&]( int off, int len ) { os << "        " << sv( len ) << " pi" << off << ";\n"; } );

    // load
    int s = 8;
    os << "\n";
    for_each_reg( [&]( int off, int len ) {
        if ( off >= size_for_test_during_load ) {
            os << std::string( s, ' ' ) << "if ( nodes_size > " << off << " ) {\n";
            s += 4;
        }
        os << std::string( s, ' ' ) << "px" << off << " = " << fv( len ) << "::load_aligned( position_xs + " << off << " );\n";
        os << std::string( s, ' ' ) << "py" << off << " = " << fv( len ) << "::load_aligned( position_ys + " << off << " );\n";
        os << std::string( s, ' ' ) << "pi" << off << " = " << sv( len ) << "::load_aligned( cut_ids + "     << off << " );\n";
    } );
    while( ( s -= 4 ) >= 8 )
        os << std::string( s, ' ' ) << "}\n";

}

void GenPlaneCutProc::gen_store( std::ostream &os, int s ) {
    int o = s;
    for_each_reg( [&]( int off, int len ) {
        if ( off >= size_for_test_during_load ) {
            os << std::string( s, ' ' ) << "if ( nodes_size > " << off << " ) {\n";
            s += 4;
        }
        os << std::string( s, ' ' ) << fv( len ) << "::store_aligned( position_xs + " << off << ", px" << off << " );\n";
        os << std::string( s, ' ' ) << fv( len ) << "::store_aligned( position_ys + " << off << ", py" << off << " );\n";
        os << std::string( s, ' ' ) << sv( len ) << "::store_aligned( cut_ids + "     << off << ", pi" << off << " );\n";
    } );
    while( ( s -= 4 ) >= o )
        os << std::string( s, ' ' ) << "}\n";
}

void GenPlaneCutProc::for_each_reg( std::function<void( int, int )> f ) {
    for_each_reg( f, size );
}

void GenPlaneCutProc::for_each_reg( std::function<void( int, int )> f, int size ) {
    for( int len = simd_size(), off = 0; len; len /= 2 )
        for( ; off + len <= size; off += len )
            f( off, len );
}

int GenPlaneCutProc::scalar_size() {
    if ( scalar_type == "FP64" )
        return 64;
    TODO;
    return 0;
}

int GenPlaneCutProc::simd_size() {
    if ( arch == "NOARCH" )
        return 1;
    if ( arch == "AVX512" )
        return 512 / scalar_size();
    if ( arch == "AVX2" )
        return 256 / scalar_size();
    if ( arch == "SSE2" )
        return 128 / scalar_size();
    TODO;
    return 0;
}

std::string GenPlaneCutProc::fv( int len ) {
    return "SimdVec<" + scalar_type + "," + std::to_string( len ) + ">";
}

std::string GenPlaneCutProc::sv( int len ) {
    return "SimdVec<" + size_type + "," + std::to_string( len ) + ">";
}

void GenPlaneCutProc::gen_header( std::ostream &os ) {
    os << "#ifndef METHOD_INCLUDE\n";
    os << "#include \"../../support/type_config.h\"\n";
    os << "#include \"../../support/SimdVec.h\"\n";
    os << "using namespace sdot;\n";
    os << "using TF = " << scalar_type << ";\n";
    os << "using ST = " << size_type << ";\n";
    os << "\n";
    os << "void plane_cut( const TF **nds, const TF *nss, const ST *nis, ST nb_cuts, ST nodes_size, ST nodes_rese, TF *position_xs, TF *position_ys, TF *normal_xs, TF *normal_ys, TF *distances, ST *cut_ids ) {\n";
    os << "    ST num_cut = 0;\n";
    os << "    #endif\n";
}

void GenPlaneCutProc::gen_footer( std::ostream &os ) {
    os << "\n";
    os << "    #ifndef METHOD_INCLUDE\n";
    os << "}\n";
    os << "#endif\n";
}

void GenPlaneCutProc::gen_case( std::ostream &os, std::bitset<32> case_code ) {
    std::string sp = "                ";

    // nb_nodes
    int nb_nodes = -1;
    for( int i = 0; i < 32; ++i )
        if ( case_code[ i ] )
            nb_nodes = i;

    if ( nb_nodes < 0 ) {
        os << sp << "break;\n"; // not a possible case
        return;
    }

    // nb_outside_nodes
    int nb_outside_nodes = 0;
    for( int i = 0; i < nb_nodes; ++i )
        nb_outside_nodes += case_code[ i ];

    if ( nb_outside_nodes == 0 ) {
        os << sp << "return;\n";
        return;
    }

    if ( nb_outside_nodes == nb_nodes ) {
        os << sp << "nodes_size = 0;\n";
        os << sp << "return;\n";
        return;
    }

    //
    PlaneCutProcPlan pcpp( nb_nodes, case_code );
    ASSERT( pcpp.nb_cuts(), "should be filtered" );
    if ( pcpp.nb_cuts() != 2 ) {
        gen_store( os, 16 );
        os << sp << "break;\n";
        return;
    }

    //
    OptParm op;
    pcpp.write_code( os, sp, *this, op );
}

void GenPlaneCutProc::gen_cases( std::ostream &os ) {
    // get the codes
    std::map<std::string,std::vector<int>> cases; // code => cases
    for( int case_code = 0; case_code < ( 1 << ( size + 1 ) ); ++case_code ) {
        std::ostringstream ss;
        gen_case( ss, case_code );
        cases[ ss.str() ].push_back( case_code );
    }

    //
    std::ostringstream store_and_break_code;
    gen_store( store_and_break_code, 16 );
    store_and_break_code << "                break;\n";

    // jump code
    std::vector<int> case_nums( 1 << ( size + 1 ), 0 );
    int num_code = 0;
    for( const auto &p : cases ) {
        if ( p.first == store_and_break_code.str() ) {
            for( int c : p.second )
                case_nums[ c ] = -1;
            continue;
        }

        for( int c : p.second )
            case_nums[ c ] = num_code;
        ++num_code;
    }

    std::string sp = "            ";
    os << sp << "\n";
    os << sp << "static void *dispatch_table[] = {";
    for( std::size_t i = 0; i < case_nums.size(); ++i ) {
        if ( case_nums[ i ] >= 0 )
            os << ( i % 8 ? " " : "\n    " + sp ) << "&&case_" << case_nums[ i ] << ",";
        else
            os << ( i % 8 ? " " : "\n    " + sp ) << "&&store_and_break,";
    }
    os << "\n" << sp << "};\n";
    os << sp << "goto *dispatch_table[ case_code ];\n";

    // cases code
    num_code = 0;
    os << sp << "\n";
    for( auto p : cases ) {
        if ( p.first == store_and_break_code.str() )
            continue;
        os << sp << "case_" << num_code << ": {\n";
        os << p.first;
        os << sp << "}\n";
        ++num_code;
    }

    os << sp << "store_and_break: {\n";
    gen_store( os, 16 );
    os << "                break;\n";
    os << sp << "}\n";
}




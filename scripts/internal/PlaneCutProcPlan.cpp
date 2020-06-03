#include "../../src/sdot/support/OptParm.h"
#include "../../src/sdot/support/ERROR.h"
#include "../../src/sdot/support/TODO.h"
#include "../../src/sdot/support/P.h"
#include "PlaneCutProcPlan.h"
#include "GenPlaneCutProc.h"
#include "SimdGen.h"
#include <ostream>

PlaneCutProcPlan::PlaneCutProcPlan( int old_nb_nodes, std::bitset<32> outside_nodes ) : outside_nodes( outside_nodes ), old_nb_nodes( old_nb_nodes ) {
    for( int i = 0; i < old_nb_nodes; ++i ) {
        int j = ( i + 1 ) % old_nb_nodes;

        // inside => inside
        if ( outside_nodes[ i ] == false && outside_nodes[ j ] == false ) {
            new_nodes.push_back( { i, -1 } );
            continue;
        }

        // inside => outside
        if ( outside_nodes[ i ] == false && outside_nodes[ j ] == true ) {
            new_nodes.push_back( { i, -1 } );
            new_nodes.push_back( { i, j } );
            continue;
        }

        // outside => inside
        if ( outside_nodes[ i ] == true && outside_nodes[ j ] == false ) {
            new_nodes.push_back( { i, j } );
            continue;
        }

        // outside => outside
        if ( outside_nodes[ i ] == true && outside_nodes[ j ] == true ) {
            continue;
        }
    }
}

void PlaneCutProcPlan::write_code( std::ostream &os, const std::string &sp, GenPlaneCutProc &gp, OptParm &/*op*/ ) {
    // info (comment)
    os << sp << "// old_nb_nodes: " << old_nb_nodes << " nodes:";
    for( const Item &item : new_nodes ) {
        if ( item.n1 < 0 )
            os << " " << item.n0;
        else
            os << " [" << item.n0 << "," << item.n1 << "]";
    }
    os << "\n";

    // size
    if ( new_nodes.size() != std::size_t( old_nb_nodes ) )
        os << sp << "nodes_size = " << new_nodes.size() << ";\n";

    //
    std::vector<int> ci = cut_indices();
    if ( ci.size() != 2 )
        ERROR( "should be filtered" );

    // SimdGen handles SimdVec<...> only
    os << sp << "const SimdVec<" << gp.size_type << ",1> *snis = reinterpret_cast<const SimdVec<" << gp.size_type << ",1> *>( nis );\n";

    //
    SimdGen sg;
    sg.arch = gp.arch;

    // input values
    using ST = SimdGen::ST;
    std::vector<ST> dis, pxs, pys, pis;
    for( auto t : { std::make_pair( "di", &dis ), std::make_pair( "px", &pxs ), std::make_pair( "py", &pys ), std::make_pair( "pi", &pis ) } ) {
        gp.for_each_reg( [&]( int off, int len ) {
            ST var = sg.new_var( std::string( t.first ) + std::to_string( off ), len, std::string( t.first ) == "pi" ? gp.size_type : gp.scalar_type );
            for( ST sub = 0; sub < ST( len ); ++sub )
                t.second->push_back( sg.new_gather( { var }, { sub } ) );
        } );
    }

    for( auto t : { std::make_pair( "x", &pxs ), std::make_pair( "y", &pys ), std::make_pair( "i", &pis ) } ) {
        std::string nc = t.first;
        std::vector<ST> &p_s = *t.second;
        std::string type = nc == "i" ? gp.size_type : gp.scalar_type;
        auto wrp = [&]( int off, int len ) {
            std::vector<ST> nops, nouts;
            for( int sub = 0; sub < len; ++sub ) {
                nouts.push_back( 0 );
                int ind = off + sub;


                //
                if ( std::size_t( ind ) >= new_nodes.size() ) {
                    nops.push_back( sg.new_undefined( 1, type ) );
                    continue;
                }

                //
                const Item &item = new_nodes[ ind ];
                if ( item.n1 >= 0 ) {
                    if ( nc == "i" ) {
                        if ( outside_nodes[ item.n1 ] )
                            nops.push_back( sg.new_var( "snis[ num_cut ]", 1, gp.size_type ) );
                        else
                            nops.push_back( p_s[ item.n0 ] );
                        continue;
                    }

                    // px0 + di1 / ( di1 - di0 ) * ( px1 - px0 )
                    nops.push_back( sg.new_add(
                        p_s[ item.n0 ],
                        sg.new_mul(
                            sg.new_div( dis[ item.n0 ], sg.new_sub( dis[ item.n0 ], dis[ item.n1 ] ) ),
                            sg.new_sub( p_s[ item.n1 ], p_s[ item.n0 ] )
                        )
                    ) );
                    continue;
                }

                //
                nops.push_back( p_s[ item.n0 ] );
            }
            std::string out_name = std::string( "p" ) + nc + std::to_string( off );
            if ( off >= gp.size )
                out_name = std::string( "reinterpret_cast<SimdVec<" ) + ( nc == "i" ? gp.size_type : gp.scalar_type ) + ",1> *>( " + ( nc == "i" ? "cut_ids" : "position_" + nc + "s" ) + " )" + "[ " + std::to_string( off ) + " ]";
            sg.add_write( out_name, sg.new_gather( nops, nouts ) );
        };

        gp.for_each_reg( wrp );
        for( std::size_t i = gp.size; i < new_nodes.size(); ++i )
            wrp( i, 1 );
    }

    sg.gen_code( os, sp );

    //
    if ( new_nodes.size() > std::size_t( gp.size ) ) {
        os << sp << "++num_cut;\n";
        os << sp << "goto store_and_break;\n";
    } else
        os << sp << "continue;\n";
}

std::vector<int> PlaneCutProcPlan::cut_indices() {
    std::vector<int> res;
    for( std::size_t i = 0; i < new_nodes.size(); ++i )
        if ( new_nodes[ i ].n1 >= 0 )
            res.push_back( i );
    return res;
}

void PlaneCutProcPlan::make_svec( std::ostream &os, const std::string &sp, std::string name, std::string n0, std::string n1 ) {
    os << sp << "SimdVec<TF,2> " << name << "( " << n0;
    if ( n0 != n1 )
        os << ", " << n1;
    os << " );\n";
}

int PlaneCutProcPlan::nb_cuts() {
    return cut_indices().size();
}

std::string PlaneCutProcPlan::sval( GenPlaneCutProc &gp, std::string n, int ind ) {
    std::string res;
    gp.for_each_reg( [&]( int off, int len ) {
        if ( ind >= off && ind < off + len )
            res = n + std::to_string( off ) + "[ " + std::to_string( ind - off ) + " ]";
    } );
    return res;
}


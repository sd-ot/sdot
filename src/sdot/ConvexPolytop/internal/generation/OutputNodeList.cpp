#include "../../../support/TODO.h"
#include "../../../support/P.h"
#include "OutputNodeList.h"
#include <algorithm>
using TI = OutputNodeList::TI;

void OutputNodeList::write_to_stream( std::ostream &os ) const {
    os << "perm_src: " << perm_src_nodes;
    os << "\n  nbrs: " << nbrs;
    os << "\n  summary: " << summary();
}

void OutputNodeList::write_function_call( std::ostream &os, TI num_case, std::vector<std::string> ref_shape_names, const std::map<TI,TI> &src_node_map ) const {
    os << "    CpOps::run_";
    write_function_name( os );
    os << "( beg_cut_cases[ " << num_case << " ], nb_cut_cases[ " << num_case << " ]";
    for( TI num_nbr = 0; num_nbr < perm_nbrs.size(); ++num_nbr ) {
        const ByRefShape &nbr = nbrs[ perm_nbrs[ num_nbr ] ];
        os << ", shape_list( tmp_shape_map, \"" << ref_shape_names[ nbr.num_dst_ref_shape ] << "\" ), { ";
        write_perm( os, nbr.perm_dst_nodes );
        os << " }";
    }
    os << ", sc, { ";
    write_perm( os, make_inv_perm( perm_src_nodes, src_node_map ) );
    os << " }, N<dim>() );\n";
}

void OutputNodeList::write_perm( std::ostream &os, const std::vector<TI> &perm ) const {
    for( TI n = 0; n < perm.size(); ++n )
        os << ( n ? ", " : "" ) << perm[ n ];

}

std::vector<TI> OutputNodeList::make_inv_perm( const std::vector<TI> &perm, const std::map<TI,TI> &src_node_map ) const {
    std::vector<TI> inv_perm( perm.size() );
    for( auto p : src_node_map )
        if ( p.second < perm.size() )
            inv_perm[ perm[ p.second ] ] = p.first;
    return inv_perm;
}

void OutputNodeList::write_function_name( std::ostream &os ) const {
    for( TI num_nbr = 0, cs = 0; num_nbr < perm_nbrs.size(); ++num_nbr ) {
        const ByRefShape &nbr = nbrs[ perm_nbrs[ num_nbr ] ];
        for( TI num_dst_shape : nbr.perm_dst_shapes ) {
            if ( cs++ )
                os << "__";
            os << num_nbr;
            for( std::pair<TI,TI> p : summary( nbr.node_lists[ num_dst_shape ], nbr.perm_dst_nodes ) )
                os << "_" << p.first << "_" << p.second;
        }
    }

}

void OutputNodeList::sort_with_first_shape_proposal( ByRefShape &nbr ) {
    // sort dst nodes / first dst shape
    std::sort( nbr.perm_dst_nodes.begin(), nbr.perm_dst_nodes.end(), [&]( TI a, TI b ) {
        return summary( nbr.node_lists[ nbr.perm_dst_shapes[ 0 ] ][ a ] ) <
                summary( nbr.node_lists[ nbr.perm_dst_shapes[ 0 ] ][ b ] );
    } );
    // sort dst shapes
    std::sort( nbr.perm_dst_shapes.begin(), nbr.perm_dst_shapes.end(), [&]( TI a, TI b ) {
        return summary( nbr.node_lists[ a ], nbr.perm_dst_nodes ) <
                summary( nbr.node_lists[ b ], nbr.perm_dst_nodes );
    } );
}

void OutputNodeList::sort_with_fixed_src_node_perm() {
    for( ByRefShape& nbr : nbrs ) {
        std::vector<TI> perm_dst_shapes = nbr.perm_dst_shapes;
        sort_with_first_shape_proposal( nbr );

        for( TI first_shape = 1; first_shape < nbr.node_lists.size(); ++first_shape ) {
            std::swap( perm_dst_shapes[ 0 ], perm_dst_shapes[ first_shape ] );

            ByRefShape trial_nbr = nbr;
            trial_nbr.perm_dst_shapes = perm_dst_shapes;
            sort_with_first_shape_proposal( trial_nbr );

            if ( summary( nbr ) > summary( trial_nbr ) )
                nbr = trial_nbr;
        }
    }

    // sort nbrs
    std::sort( perm_nbrs.begin(), perm_nbrs.end(), [&]( TI a, TI b ) {
        return summary( nbrs[ a ] ) <
               summary( nbrs[ b ] );
    } );
}

std::pair<TI,TI> OutputNodeList::summary( const std::pair<TI,TI> &src_nodes ) const { // [ num_dst_shape ][ num_dst_node ]
    TI n0 = perm_src_nodes[ src_nodes.first ];
    TI n1 = perm_src_nodes[ src_nodes.second ];
    return { std::min( n0, n1 ), std::max( n0, n1 ) };
}

std::vector<std::pair<TI, TI> > OutputNodeList::summary( const std::vector<std::pair<TI,TI>> &src_node_list, const std::vector<TI> &perm_dst_nodes ) const { // [ num_dst_shape ][ num_dst_node ]
    std::vector<std::pair<TI,TI>> res;
    for( TI j : perm_dst_nodes )
        res.push_back( summary( src_node_list[ j ] ) );
    return res;
}

std::vector<std::vector<std::pair<TI, TI> > > OutputNodeList::summary( const ByRefShape &nbr ) const { // [ num_dst_shape ][ num_dst_node ]
    std::vector<std::vector<std::pair<TI,TI>>> res;
    for( TI i : nbr.perm_dst_shapes )
        res.push_back( summary( nbr.node_lists[ i ], nbr.perm_dst_nodes ) );
    return res;
}

std::vector<std::vector<std::vector<std::pair<TI, TI> > > > OutputNodeList::summary() const { // [ num_dst_ref_shape ][ num_dst_shape ][ num_dst_node ]
    std::vector<std::vector<std::vector<std::pair<TI,TI>>>> res;
    for( const ByRefShape &nbr : nbrs )
        res.push_back( summary( nbr ) );
    return res;
}

void OutputNodeList::ByRefShape::write_to_stream( std::ostream &os ) const {
    os << "num_dst_ref_shape: " << num_dst_ref_shape;
    os << " perm_dst_nodes: "   << perm_dst_nodes;
    os << " perm_dst_shapes: "  << perm_dst_shapes;
    os << " node_lists: "       << node_lists;

}

#include "../support/generic_ostream_output.h"
#include "CutOpWithNamesAndInds.h"
#include <sstream>
#include <set>

CutOpWithNamesAndInds::CutOpWithNamesAndInds( TI dim ) {
    cut_op.dim = dim;
}

void CutOpWithNamesAndInds::for_each_new_edge( std::function<void( TI n00, TI n01, TI n10, TI n11 )> f ) const {
    for( const CutItem &cut_item : cut_op.cut_items ) {
        for( std::size_t num_node = 0; num_node < cut_item.nodes.size(); ++num_node ) {
            if ( cut_item.faces[ num_node ] == TI( CutItem::cut_face ) ) {
                std::size_t mum_node = ( num_node + 1 ) % cut_item.nodes.size();
                f(
                    input_node_inds[ cut_item.nodes[ num_node ][ 0 ] ],
                    input_node_inds[ cut_item.nodes[ num_node ][ 1 ] ],
                    input_node_inds[ cut_item.nodes[ mum_node ][ 0 ] ],
                    input_node_inds[ cut_item.nodes[ mum_node ][ 1 ] ]
                );
            }
        }
    }
}

std::size_t CutOpWithNamesAndInds::nb_created( std::string shape_name ) const {
    std::size_t res = 0;
    for( const Out &output : outputs )
        res += output.shape_name == shape_name;
    return res;
}

std::string CutOpWithNamesAndInds::created_shapes() const {
    std::set<std::string> so;
    for( std::size_t num_out = 0; num_out < outputs.size(); ++num_out ) {
        std::ostringstream ss;
        ss << "S: " << outputs[ num_out ].shape_name;
        ss << " N: " << cut_op.cut_items[ num_out ].nodes;
        ss << " F: " << cut_op.cut_items[ num_out ].faces;
        ss << " L: " << cut_op.cut_items[ num_out ].lengths;
        so.insert( ss.str() );
    }

    std::string res;
    for( std::string s : so )
        res += ( res.size() ? " " : "" ) + s;
    return res;
}


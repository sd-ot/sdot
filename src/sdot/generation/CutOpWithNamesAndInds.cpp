#include "../support/generic_ostream_output.h"
#include "CutOpWithNamesAndInds.h"
#include <sstream>
#include <set>

namespace sdot {

std::size_t CutOpWithNamesAndInds::nb_created( std::string shape_name ) const {
    std::size_t res = 0;
    for( const Out &output : outputs )
        res += output.shape_name == shape_name;
    return res;
}

sdot::CutOpWithNamesAndInds::CutOpWithNamesAndInds( TI dim ) {
    cut_op.dim = dim;
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

} // namespace sdot

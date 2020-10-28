#include "CutOpWithNamesAndInds.h"

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

} // namespace sdot

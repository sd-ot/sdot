#include "GlobGeneGeomData.h"

namespace sdot {

std::string GlobGeneGeomData::mk_item_name( std::vector<TI> inds ) {
    needed_cut_ops.insert( inds );

    std::string res = "mk_items";
    for( TI i : inds )
        res += "_" + std::to_string( i );
    return res;
}

}


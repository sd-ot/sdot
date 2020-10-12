#include "SetOfElementaryPolytops.h"
#include "../support/ASSERT.h"
#include "../support/P.h"

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( KernelSlot *ks, unsigned dim ) : dim( dim ), ks( ks ) {
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os, const std::string &sp ) const {
    os << sp << "SetOfElementaryPolytops([";
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;
        os << "\n" << sp << "  " << sd.shape_type->name();
        for( std::size_t i = 0; i < sd.size(); ++i ) {
            os << "\n" << sp << "   ";
            for( std::size_t d = 0; d < sd.coordinates.size(); ++d )
                sd.coordinates[ d ].display( os << " ", i, 1 );
        }
    }
    os << "\n" << sp << "])";
}

void SetOfElementaryPolytops::add_repeated( ShapeType *shape_type, SetOfElementaryPolytops::BI count, const VecTF &coordinates ) {
    ASSERT( coordinates.size() == dim * shape_type->nb_nodes(), "wrong coordinates size" );
    ShapeData *sd = shape_data( shape_type );

    BI os = sd->size();
    sd->resize( os + count );

    for( std::size_t i = 0; i < coordinates.size(); ++i )
        ks->assign_repeated_TF( sd->coordinates[ i ].data(), os, coordinates.data(), i, count );
}

ShapeData *SetOfElementaryPolytops::shape_data( ShapeType *shape_type ) {
    auto iter = shape_map.find( shape_type );
    if ( iter == shape_map.end() )
        iter = shape_map.insert( iter, { shape_type, ShapeData{ ks, shape_type, dim } } );
    return &iter->second;
}


}

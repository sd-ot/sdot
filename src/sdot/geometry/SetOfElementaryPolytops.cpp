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

void SetOfElementaryPolytops::display_vtk( VtkOutput &vo ) const {
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;

        std::vector<std::tuple<const void *,BI,BI>> tfs;
        std::vector<std::tuple<const void *,BI,BI>> tis;
        for( const VecTF &c : sd.coordinates )
            tfs.emplace_back( c.data(), 0, c.size() );

        ks->get_local( [&]( const double **tfs, const BI **tis ) {
            sd.shape_type->display_vtk( vo, tfs, tis, dim, sd.size() );
        }, tfs.data(), tfs.size(), tis.data(), tis.size() );
    }
    //    std::vector<VecTF> coordinates; ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    //    ShapeType*         shape_type;  ///<
    //    std::vector<VecTI> face_ids;    ///< all the ids for node 0, all the ids for node 1, ...
    //    VecTI              ids;         ///<
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

#include <parex/containers/kernels/init_vector_from_value.h>
#include <parex/containers/kernels/init_vector_from_iota.h>
#include "HomogeneousElementaryPolytopList.h"

using namespace parex;

template<class SM,class Coordinates,class FaceIds>
void add_repeated( SM &shape_map, const std::string &shape_name, std::size_t count, const Coordinates &coordinates, const FaceIds &face_ids, std::size_t beg_ids ) {
    using TI = std::size_t;
    shape_map.apply_on( shape_name, [&]( auto &hl ) {
        auto proc = hl.positions.default_processor();
        TI os = hl.size(), ns = os + count;
        hl.resize( ns, proc );

        for( TI i = 0; i < hl.nb_nodes; ++i )
            for( TI d = 0; d < hl.dim; ++d )
                parex::init_vector_from_value( proc, hl.positions.data( i, d, os ), *coordinates.data( i, d ), count );
        for( TI i = 0; i < face_ids.size(); ++i )
            parex::init_vector_from_value( proc, hl.face_ids.data( i, os ), *face_ids.data( i ), count );
        parex::init_vector_from_iota( proc, hl.ids.data( os ), beg_ids, count );
    } );
}

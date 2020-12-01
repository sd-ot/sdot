#include "HomogeneousElementaryPolytopList.h"

#include <asimd/operations/assign_scalar.h>
#include <asimd/operations/assign_iota.h>

#include <parex/generic_ostream_output.h>
#include <parex/TaskOut.h>
#include <parex/P.h>

#include <sstream>

template<class TF,class TI,class TC,class A,class B,class TB>
void add_repeated_( HomogeneousElementaryPolytopList<TF,TI> &hl, TC count, const A &coordinates, const B &face_ids, TB beg_ids ) {
    TI os = hl.size(), ns = os + count;
    hl.resize( ns );

    for( TI i = 0; i < hl.nb_nodes(); ++i )
        for( TI d = 0; d < hl.dim(); ++d )
            asimd::assign_scalar( &hl.positions( i, d, os ), coordinates( i, d ), count );
    for( TI i = 0; i < face_ids.size(); ++i )
        asimd::assign_scalar( &hl.face_ids( i, os ), face_ids( i ), count );
    asimd::assign_iota( &hl.ids( os ), beg_ids, count );
}

template<class ShapeMap,class TI,class A,class B,class C>
TaskOut<ShapeMap> add_repeated( TaskOut<ShapeMap> &shape_map, TaskOut<std::string> &shape_name, TaskOut<TI> &count, TaskOut<A> &coordinates, TaskOut<B> &face_ids, TaskOut<C> &beg_ids ) {
    add_repeated_( *(*shape_map)[ *shape_name ], *count, *coordinates, *face_ids, *beg_ids );
    return std::move( shape_map );
}

#include <sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/ShapeMap.h>
#include <parex/arch/assign_repeated.h>
#include <parex/arch/assign_iota.h>
#include <parex/support/P.h>
#include <parex/TaskRef.h>

using namespace parex;
using namespace sdot;

template<class TF,class TI,int dim,class A,class B,class C>
ShapeMap<TF,TI,dim> *add_repeated(
    Task *task, ShapeMap<TF,TI,dim> &shape_map, const std::string &shape_name, const ElementaryPolytopOperations &eto,
    TI count, const A &input_coordinates, const B &input_face_ids, C beg_ids ) {

    if ( ! task->move_arg( 0 ) )
        ERROR( "shape_map not owned" );

    ShapeData<TF,TI,dim> &sd = shape_map.shape_data( shape_name, eto );
    TI old_size = sd.size;
    sd.size += count;

    sd.coordinates.resize( sd.size );
    sd.face_ids.resize( sd.size );
    sd.ids.resize( sd.size );

    for( TI i = 0; i < input_coordinates.size(); ++i )
        assign_repeated( sd.coordinates.ptr( i ), old_size, sd.size, input_coordinates[ i ] );
    for( TI i = 0; i < input_face_ids.size(); ++i )
        assign_repeated( sd.face_ids.ptr( i ), old_size, sd.size, input_face_ids[ i ] );
    assign_iota<TI>( sd.ids.data(), old_size, sd.size, beg_ids );

    return nullptr;
}

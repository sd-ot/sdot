#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>

#include <parex/arch/assign_repeated.h>
#include <parex/arch/assign_iota.h>

#include <parex/TaskRef.h>

using namespace parex;

template<class TF,class TI,class A,class B,class C>
std::tuple<Tensor<TF>*, Tensor<TI>*, Vec<TI>*> add_repeated_elements(
    Task *task, Tensor<TF> &coordinates, Tensor<TI> &face_ids, Vec<TI> &ids,
    TI count, const A &input_coordinates, const B &input_face_ids, C beg_ids ) {

    if ( ! task->move_arg( { 0, 1, 2 }, { 0, 1, 2 } ) )
        ERROR( "not owned" );

    TI old_size = ids.size(), new_size = old_size + count;
    coordinates.resize( new_size );
    face_ids.resize( new_size );
    ids.resize( new_size );

    for( TI i = 0; i < input_coordinates.size(); ++i )
        assign_repeated( coordinates.ptr( i ), old_size, new_size, input_coordinates[ i ] );
    for( TI i = 0; i < input_face_ids.size(); ++i )
        assign_repeated( face_ids.ptr( i ), old_size, new_size, input_face_ids[ i ] );
    assign_iota<TI>( ids.data(), old_size, new_size, beg_ids );

    return { nullptr, nullptr, nullptr };
}

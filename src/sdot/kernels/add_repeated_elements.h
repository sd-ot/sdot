#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>

#include <parex/arch/assign_repeated.h>
#include <parex/arch/assign_iota.h>

#include <parex/support/P.h>
#include <parex/TaskRef.h>
#include <cstdint>
#include <random>

using namespace parex;

template<class TF,class TI,class A,class B,class C>
std::tuple<Tensor<TF>*, Tensor<TI>*, Vec<TI>*> add_repeated_elements(
    Task *task, Tensor<TF> &coordinates, Tensor<TI> &face_ids, Vec<TI> &ids,
    TI count, const A &input_coordinates, const B &input_face_ids, C beg_ids ) {

    P( coordinates.size );

    //
    if ( ! task->move_arg( { 0, 1, 2 }, { 0, 1, 2 } ) )
        ERROR( "not owned" );

    TI old_size = ids.size(), new_size = old_size + count;
    coordinates.resize( new_size );
    face_ids.resize( new_size );
    ids.resize( new_size );

    //    SimdRange<SimdSize<TF>::value>::for_each_iota( count, [&]( TI index, auto iota, auto s ) {
    //        using SV = SimdVec<TF,s.value>;
    //        SV::store_aligned( ids.data() + old_size + index, iota );
    //    }, beg_ids );
    for( TI i = 0; i < input_face_ids.size(); ++i )
        assign_repeated( face_ids.data( i ), input_face_ids[ i ], old_size, new_size );
    assign_iota<TI>( ids.data(), old_size, new_size, beg_ids );
    P( ids );

    return { nullptr, nullptr, nullptr };
}

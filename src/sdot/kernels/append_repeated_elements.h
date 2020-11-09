#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>
#include <parex/support/P.h>
#include <parex/Task.h>
#include <cstdint>
#include <random>

using namespace parex;

template<class TF,class TI>
std::tuple<Tensor<TF>*, Tensor<TI>*, Vec<TI>*> append_repeated_elements(
    Task *task, Tensor<TF> &coordinates, Tensor<TI> &face_ids, Vec<TI> &ids,
    TI count, const Tensor<TF> &input_coordinates, const Vec<TF> &input_face_ids, TI beg_ids ) {

    //
    for( std::size_t i = 0; i < 3; ++i )
        if ( ! task->move_arg( i, i )  )
            ERROR( "" );

    P( coordinates.size );

    return { nullptr, nullptr, nullptr };
}

#include "ShapeData.h"
#include <map>

using namespace parex;

template<class TF,class TI,int dim>
std::map<std::string,ShapeData<TF,TI,dim>> *new_shape_map(  ) {

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

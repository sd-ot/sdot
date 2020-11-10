#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>
#include <sstream>
using namespace parex;

template<class TF,class TI>
std::string *shape_data_to_string( Tensor<TF> &coordinates, Tensor<TI> &face_ids, Vec<TI> &ids, const std::string &sp ) {
    std::ostringstream ss;
    for( TI i = 0; i < ids.size(); ++i ) {
        ss << "\n" << sp << "C:";
        for( TI c = 0; c < coordinates.nb_x_vec(); ++c )
            ss << " " << std::setw( 8 ) << coordinates.ptr( c )[ i ];
        ss << " F:";
        for( TI c = 0; c < face_ids.nb_x_vec(); ++c )
            ss << " " << std::setw( 6 ) << face_ids.ptr( c )[ i ];
        ss << " I: " << ids[ i ];
    }

    return new std::string( ss.str() );
}

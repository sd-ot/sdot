#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>
#include <parex/support/P.h>

#include "../geometry/VtkOutput.h"

using namespace parex;
using namespace sdot;

template<class TF,class TI,int dim,class VN>
void shape_data_display_vtk( void *vtk_output_, void *offsets_, Tensor<TF> &coordinates, Tensor<TI> &/*face_ids*/, Vec<TI> &ids, N<dim>, unsigned vtk_id, const VN &nodes ) {
    VtkOutput &vo = *reinterpret_cast<VtkOutput *>( vtk_output_ );
    VtkOutput::Pt *offsets = reinterpret_cast<VtkOutput::Pt *>( offsets_ );

    std::vector<VtkOutput::Pt> pts( nodes.size(), VtkOutput::Pt( 0.0 ) );
    if ( offsets ) {
        for( TI i = 0; i < ids.size(); ++i ) {
            for( TI c = 0; c < nodes.size(); ++c )
                for( int d = 0; d < dim; ++d )
                    pts[ c ][ d ] = coordinates.ptr( nodes[ c ] * dim + d )[ i ] + offsets[ ids[ i ] ][ d ];
            vo.add_item( pts.data(), pts.size(), vtk_id );
        }
    } else {
        for( TI i = 0; i < ids.size(); ++i ) {
            for( TI c = 0; c < nodes.size(); ++c )
                for( int d = 0; d < dim; ++d )
                    pts[ c ][ d ] = coordinates.ptr( nodes[ c ] * dim + d )[ i ];
            vo.add_item( pts.data(), pts.size(), vtk_id );
        }
    }
}

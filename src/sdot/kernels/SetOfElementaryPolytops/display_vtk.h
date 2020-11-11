#include <sdot/geometry/VtkOutput.cpp>
#include <sdot/geometry/ShapeMap.h>
#include <parex/support/P.h>

using namespace parex;
using namespace sdot;

template<class TF,class TI,int dim>
void display_vtk( ShapeMap<TF,TI,dim> &shape_map, const std::string &filename ) {
    VtkOutput vo;

    for( const auto &p : shape_map.map ) {
        const ShapeData<TF,TI,dim> &sd = p.second;

        p.first->display_vtk( [&]( unsigned vtk_id, const parex::Vec<unsigned> &nodes ) {
            std::vector<VtkOutput::Pt> pts( nodes.size(), VtkOutput::Pt( 0.0 ) );

            for( TI i = 0; i < sd.ids.size(); ++i ) {
                for( TI c = 0; c < nodes.size(); ++c )
                    for( int d = 0; d < dim; ++d )
                        pts[ c ][ d ] = sd.coordinates.ptr( nodes[ c ] * dim + d )[ i ];
                vo.add_item( pts.data(), pts.size(), vtk_id );
            }
        } );
    }

    vo.save( filename );
}

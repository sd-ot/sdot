#include <sdot/geometry/ElementaryPolytopOperations.h>
#include <sdot/geometry/VtkOutput.cpp>
#include <sdot/geometry/ShapeMap.h>
#include <parex/support/P.h>

using namespace parex;
using namespace sdot;

template<class TF,class TI,int dim>
void display_vtk( const std::string &filename, const ShapeMap<TF,TI,dim> &shape_map, const ElementaryPolytopOperations &epo ) {
    VtkOutput vo( { "id" } );

    std::vector<VtkOutput::TF> ids( 1 );
    for( const auto &p : shape_map.map ) {
        const ShapeData<TF,TI,dim> &sd = p.second;
        const ElementaryPolytopOperations::Operations &op = epo.operation_map.find( p.first )->second;
        for( const std::pair<TI,std::vector<TI>> &ve : op.vtk_elements ) { // [vtk_id + [node numbers]]
            std::vector<VtkOutput::Pt> pts( ve.second.size(), VtkOutput::Pt( 0.0 ) );

            for( TI i = 0; i < sd.ids.size(); ++i ) {
                ids[ 0 ] = sd.ids[ i ];

                for( TI c = 0; c < ve.second.size(); ++c )
                    for( int d = 0; d < dim; ++d )
                        pts[ c ][ d ] = sd.coordinates.ptr( ve.second[ c ] * dim + d )[ i ];

                vo.add_item( pts.data(), pts.size(), ve.first, ids );
            }
        }
    }

    vo.save( filename );
}

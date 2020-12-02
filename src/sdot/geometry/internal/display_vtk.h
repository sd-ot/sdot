#include "HomogeneousElementaryPolytopList.h"
#include "ElementaryPolytopInfoListContent.h"
#include "../VtkOutput.cpp"

#include <parex/TaskOut.h>

using namespace sdot;

template<class ShapeMap>
void display_vtk( TaskOut<std::string> &filename, TaskOut<ShapeMap> &shape_map, TaskOut<ElementaryPolytopInfoListContent> &epil ) {
    VtkOutput vo( { "id" } );

    std::vector<VtkOutput::TF> ids( 1 );
    for( const ElementaryPolytopInfo &epi : epil->elem_info ) {
        const auto *hl = shape_map->sub_list( epi.name );
        for( const std::pair<unsigned,std::vector<unsigned>> &ve : epi.vtk_elements ) { // [vtk_id + [node numbers]]
            std::vector<VtkOutput::Pt> pts( ve.second.size(), VtkOutput::Pt( 0.0 ) );
            for( unsigned num_item = 0; num_item < hl->size(); ++num_item ) {
                ids[ 0 ] = hl->ids.at( shape_map->allocator_TI, num_item );

                for( unsigned c = 0; c < ve.second.size(); ++c )
                    for( unsigned d = 0; d < hl->dim(); ++d )
                        pts[ c ][ d ] = hl->positions.at( shape_map->allocator_TF, ve.second[ c ], d, num_item );

                vo.add_item( pts.data(), pts.size(), ve.first, ids );
            }
        }
    }

    vo.save( *filename );
}

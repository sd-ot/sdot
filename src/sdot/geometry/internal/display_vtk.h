#include "HomogeneousElementaryPolytopList.h"
#include "ElementaryPolytopCaracList.h"
#include "../VtkOutput.cpp"
#include <parex/utility/P.h>

using namespace sdot;

template<class ShapeMap>
void display_vtk( const std::string &filename, ShapeMap &shape_map, const ElementaryPolytopCaracList &epcl ) {
    VtkOutput vo( { "id" } );

    std::vector<VtkOutput::TF> ids( 1 );
    for( const ElementaryPolytopCarac &epc : epcl.elements ) {
        shape_map.apply_on( epc.name, [&]( const auto &hl ) {
            parex::get_local( hl.positions.default_processor(), hl.positions.data(), hl.positions.nb_reserved_items(), [&]( auto position_ptr ) {
                //parex::get_local( hl.face_ids.default_processor(), hl.face_ids.data(), hl.face_ids.nb_reserved_items(), [&]( auto face_id_ptr ) {
                parex::get_local( hl.ids.default_processor(), hl.ids.data(), hl.ids.nb_reserved_items(), [&]( auto id_ptr ) {
                    for( const std::pair<unsigned,std::vector<unsigned>> &ve : epc.vtk_elements ) { // [vtk_id + [node numbers]]
                        P( ve.first );
                        std::vector<VtkOutput::Pt> pts( ve.second.size(), VtkOutput::Pt( 0.0 ) );
                        for( unsigned num_item = 0; num_item < hl.size(); ++num_item ) {
                            ids[ 0 ] = id_ptr[ hl.ids.offset( num_item ) ];

                            for( unsigned c = 0; c < ve.second.size(); ++c )
                                for( unsigned d = 0; d < hl.dim; ++d )
                                    pts[ c ][ d ] = position_ptr[ hl.positions.offset( ve.second[ c ], d, num_item ) ];

                            vo.add_item( pts.data(), pts.size(), ve.first, ids );
                        }
                    }
                } );
                //} );
            } );
        } );
    }


    vo.save( filename );
}

#include "SetOfElementaryPolytops.h"
#include <parex/support/ASSERT.h>
#include <parex/support/TODO.h>
#include <parex/support/P.h>

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( unsigned dim, std::string scalar_type, std::string index_type ) : scalar_type( scalar_type ), index_type( index_type ), dim( dim ) {
    // Prop: on démarre
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os, const std::string &sp ) const {
    os << sp << "SetOfElementaryPolytops([";
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;
        os << "\n" << sp << "  " << sd.shape_type->name();
        os << "\n" << sp << "   " << sd.coordinates;
        os << "\n" << sp << "   " << sd.face_ids;
        os << "\n" << sp << "   " << sd.ids;
    }
    os << "\n" << sp << "])";
}

//void SetOfElementaryPolytops::display_vtk( VtkOutput &vo, VtkOutput::Pt *offsets ) const {
//    for( const auto &p : shape_map ) {
//        const ShapeData &sd = p.second;

//        std::vector<std::tuple<const void *,BI,BI>> tfs;
//        std::vector<std::tuple<const void *,BI,BI>> tis;
//        for( std::size_t d = 0; d < sd.shape_type->nb_nodes() * dim; ++d )
//            tfs.emplace_back( sd.coordinates, sd.rese * d, sd.size );
//        tis.emplace_back( sd.ids, 0, sd.size );

//        ks->get_local( [&]( const double **tfs, const BI **tis ) {
//            sd.shape_type->display_vtk( vo, tfs, tis, dim, sd.size, offsets );
//        }, tfs.data(), tfs.size(), tis.data(), tis.size() );
//    }
//}

void SetOfElementaryPolytops::add_repeated( ShapeType *shape_type, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
    //    ASSERT( coordinates.size() == dim * shape_type->nb_nodes(), "wrong coordinates size" );
    ShapeData *sd = shape_data( shape_type );
    std::tie{ sd->coordinates, sd->face_ids, sd->ids } = Task::call( "append_repeated_elements", {
        sd->coordinates, sd->face_ids, sd->ids,
        count, coordinates, face_ids, beg_ids
    }, { 0, 1, 2 } );
}

//void SetOfElementaryPolytops::plane_cut( const std::vector<VecTF> &normals, const VecTF &scalar_products, const VecTI &cut_ids ) {
//    // conversion of normals pointers to void pointers
//    std::vector<const void *> normals_data( normals.size() );
//    for( BI i = 0; i < normals.size(); ++i )
//        normals_data[ i ] = normals[ i ].data();

//    // get scalar product, cases and new_item_count
//    std::map<const ShapeType *,BI> new_item_count;
//    for( const auto &p : shape_map ) {
//        const ShapeData &sd = p.second;

//        // reservation
//        BI nb_scalar_products = sd.rese * sd.shape_type->nb_nodes();
//        BI nb_cases = 1u << sd.shape_type->nb_nodes();
//        BI nb_offsets = nb_cases * ks->nb_lanes_TF();

//        sd.cut_out_scps = ks->allocate_TF( nb_scalar_products ); // scalar products for each element
//        sd.cut_indices  = ks->allocate_TI( sd.size ); // num elem, sorted by case number

//        void *cut_cases = ks->allocate_TI( sd.size ); // cut case for each element
//        void *offsets = ks->allocate_TI( nb_offsets ); // nb element for each cut case and for each thread

//        // get scalar products, cut_cases and counts
//        for_nb_nodes_and_dim( sd.shape_type->nb_nodes(), dim, [&]( auto nn, auto nd ) { ks->get_cut_cases(
//            cut_cases, offsets, sd.cut_out_scps, sd.coordinates, sd.ids, sd.rese,
//            normals_data.data(), scalar_products.data(), sd.size, nn, nd
//        ); } );

//        // transform counts to offsets (scan)
//        ks->count_to_offsets( offsets, sd.shape_type->nb_nodes() );

//        // get sd.cut_case_offsets
//        std::vector<BI> cut_poss_count = sd.shape_type->cut_poss_count();
//        sd.cut_case_offsets.clear();
//        sd.cut_case_offsets.resize( cut_poss_count.size() );

//        std::vector<BI> loc_cut_case_offsets( nb_cases );
//        ks->read_TI( loc_cut_case_offsets.data(), offsets, 0, nb_cases );
//        for( BI n = 0; n < nb_cases; ++n ) {
//            sd.cut_case_offsets[ n ].resize( cut_poss_count[ n ] + 1, n + 1 < nb_cases ? loc_cut_case_offsets[ n + 1 ] : sd.size );
//            sd.cut_case_offsets[ n ][ 0 ] = loc_cut_case_offsets[ n ];
//        }

//        // make indices
//        ks->sorted_indices( sd.cut_indices, offsets, cut_cases, sd.size, sd.shape_type->nb_nodes() );

//        // update of nb items to create for each type
//        sd.shape_type->cut_rese( [&]( const ShapeType *shape_type, BI count ) {
//            auto iter = new_item_count.find( shape_type );
//            if ( iter == new_item_count.end() )
//                new_item_count.insert( iter, { shape_type, count } );
//            else
//                iter->second += count;
//        }, ks, sd );

//        int cpt = 0;
//        for( auto v : sd.cut_case_offsets )
//            P( cpt++, v );

//        // free local data
//        ks->free_TI( cut_cases );
//        ks->free_TI( offsets );
//    }

//    // new shape map (using the new item counts)
//    ShapeMap old_shape_map = std::exchange( shape_map, {} );
//    for( auto p : new_item_count )
//        shape_data( p.first )->reserve( p.second );

//    //
//    for( const auto &p : old_shape_map ) {
//        const ShapeData &sd = p.second;
//        sd.shape_type->cut_ops( ks, shape_map, sd, cut_ids.data(), dim );

//        // free tmp data from old shape map
//        ks->free_TF( sd.cut_out_scps );
//        ks->free_TI( sd.cut_indices  );
//    }
//}

ShapeData *SetOfElementaryPolytops::shape_data( const ShapeType *shape_type ) {
    auto iter = shape_map.find( shape_type );
    if ( iter == shape_map.end() )
        iter = shape_map.insert( iter, { shape_type, ShapeData{ shape_type } } );
    return &iter->second;
}

}

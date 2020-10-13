#include "SetOfElementaryPolytops.h"
#include "../support/ASSERT.h"
#include "../support/TODO.h"
#include "../support/P.h"

namespace sdot {

namespace {
    template<class FU>
    void for_dim( unsigned dim, const FU &fu ) {
        #define POSSIBLE_DIM( DIM ) if ( dim == DIM ) return fu( N<DIM>() );
        #include "../kernels/possible_DIMs.h"
        #undef POSSIBLE_DIM
        TODO;
    }
}

SetOfElementaryPolytops::SetOfElementaryPolytops( KernelSlot *ks, unsigned dim ) : dim( dim ), ks( ks ) {
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os, const std::string &sp ) const {
    os << sp << "SetOfElementaryPolytops([";
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;
        os << "\n" << sp << "  " << sd.shape_type->name();
        for( std::size_t i = 0; i < sd.size; ++i ) {
            os << "\n" << sp << "   ";
            // coordinates
            for( std::size_t d = 0; d < sd.shape_type->nb_nodes() * dim; ++d )
                ks->display_TF( os << " ", sd.coordinates, sd.rese * d + i, 1 );
            // faces
            os << ",";
            for( std::size_t d = 0; d < sd.shape_type->nb_faces(); ++d )
                ks->display_TI( os << " ", sd.face_ids, sd.rese * d + i, 1 );
            // ids
            ks->display_TI( os << ", ", sd.ids, i, 1 );
        }
    }
    os << "\n" << sp << "])";
}

void SetOfElementaryPolytops::display_vtk( VtkOutput &vo ) const {
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;

        std::vector<std::tuple<const void *,BI,BI>> tfs;
        std::vector<std::tuple<const void *,BI,BI>> tis;
        for( std::size_t d = 0; d < sd.shape_type->nb_nodes() * dim; ++d )
            tfs.emplace_back( sd.coordinates, sd.rese * d, sd.size );

        ks->get_local( [&]( const double **tfs, const BI **tis ) {
            sd.shape_type->display_vtk( vo, tfs, tis, dim, sd.size );
        }, tfs.data(), tfs.size(), tis.data(), tis.size() );
    }
}

void SetOfElementaryPolytops::add_repeated( ShapeType *shape_type, SetOfElementaryPolytops::BI count, const VecTF &coordinates, const VecTI &face_ids, BI beg_ids ) {
    ASSERT( coordinates.size() == dim * shape_type->nb_nodes(), "wrong coordinates size" );
    ShapeData *sd = shape_data( shape_type );

    BI os = sd->size;
    sd->resize( os + count );

    for( std::size_t i = 0; i < sd->shape_type->nb_nodes() * dim; ++i )
        ks->assign_repeated_TF( sd->coordinates, os + i * sd->rese, coordinates.data(), i, count );
    for( std::size_t i = 0; i < sd->shape_type->nb_faces(); ++i )
        ks->assign_repeated_TI( sd->face_ids, os + i * sd->rese, face_ids.data(), i, count );
    ks->assign_iota_TI( sd->ids, os, beg_ids, count );
}

void SetOfElementaryPolytops::plane_cut( const std::vector<VecTF> &normals, const VecTF &scalar_products, const VecTI &cut_ids ) {
    // conversion of normals to void pointers
    std::vector<const void *> normals_data( normals.size() );
    for( BI i = 0; i < normals.size(); ++i )
        normals_data[ i ] = normals[ i ].data();

    // get scalar product, cases and new_item_count
    std::map<const ShapeType *,BI> new_item_count;
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;

        // reservation
        BI nb_cases = 1u << sd.shape_type->nb_nodes();
        BI nb_offsets = ks->nb_multiprocs() * nb_cases;
        sd.tmp[ ShapeData::out_scps ] = ks->allocate_TF( sd.shape_type->nb_nodes() * sd.rese );
        sd.tmp[ ShapeData::offset_0 ] = ks->allocate_TI( nb_offsets );
        sd.tmp[ ShapeData::offset_1 ] = ks->allocate_TI( nb_offsets );

        // distribute work to multiprocessors, init offsets to cases
        BI re_cases = ks->init_offsets_for_cut_cases( sd.tmp[ ShapeData::offset_0 ], sd.tmp[ ShapeData::offset_1 ], nb_cases, sd.size );
        sd.tmp[ ShapeData::cut_case ] = ks->allocate_TI( nb_cases * re_cases );

        // cut_cases
        for_dim( dim, [&]( auto nd ) { ks->get_cut_cases(
               sd.tmp[ ShapeData::cut_case ], sd.tmp[ ShapeData::offset_1 ], sd.tmp[ ShapeData::out_scps ],
               sd.coordinates, sd.ids, sd.rese, normals_data.data(), scalar_products.data(), sd.size, nd
        ); } );

        // new item count
        std::tuple<const void *,BI,BI> gos[] = { { sd.tmp[ ShapeData::offset_0 ], 0, nb_offsets }, { sd.tmp[ ShapeData::offset_1 ], 0, nb_offsets } };
        ks->get_local( [&]( const double **, const BI **offsets ) {
            sd.shape_type->cut_count( [&]( const ShapeType *shape_type, BI count ) {
                auto iter = new_item_count.find( shape_type );
                if ( iter == new_item_count.end() )
                    new_item_count.insert( iter, { shape_type, count } );
                else
                    iter->second += count;
            }, offsets );
        }, {}, 0, gos, 2 );
    }

    // new shape map
    ShapeMap old_shape_map = std::exchange( shape_map, {} );
    for( auto p : new_item_count )
        shape_data( p.first )->reserve( p.second );

    for( const auto &p : old_shape_map ) {
        const ShapeData &sd = p.second;
        sd.shape_type->cut_ops( ks, shape_map, sd, cut_ids.data(), dim );
    }

    // free tmp data from old shape map
    for( const auto &p : old_shape_map ) {
        const ShapeData &sd = p.second;
        ks->free_TF( sd.tmp[ ShapeData::out_scps ] );
        ks->free_TF( sd.tmp[ ShapeData::cut_case ] );
        ks->free_TF( sd.tmp[ ShapeData::offset_0 ] );
        ks->free_TF( sd.tmp[ ShapeData::offset_1 ] );
    }
}

ShapeData *SetOfElementaryPolytops::shape_data( const ShapeType *shape_type ) {
    auto iter = shape_map.find( shape_type );
    if ( iter == shape_map.end() )
        iter = shape_map.insert( iter, { shape_type, ShapeData{ ks, shape_type, dim } } );
    return &iter->second;
}


}

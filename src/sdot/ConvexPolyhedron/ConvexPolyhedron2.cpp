#include "../support/VtkOutput.h"
#include "../support/SimdRange.h"
#include "../support/SimdAlig.h"
#include "../support/SimdVec.h"
#include "../support/P.h"
#include "ConvexPolyhedron2.h"
#include <stdlib.h>

#define ConvexPolyhedron2 SDOT_CONCAT_TOKEN_2( ConvexPolyhedron2_, PROFILE )
namespace sdot {

ConvexPolyhedron2::ConvexPolyhedron2() {
    nodes_size = 0;
    nodes_rese = 128;
    t_nodes_rese = 128;

    // vectors linked to nodes_rese
    position_xs = allocate_TF_vec( nodes_rese );
    position_ys = allocate_TF_vec( nodes_rese );
    normal_xs   = allocate_TF_vec( nodes_rese );
    normal_ys   = allocate_TF_vec( nodes_rese );
    cut_ids     = allocate_ST_vec( nodes_rese );

    outside_nodes.resize( ( nodes_rese + 63 ) / 64 );
    distances        = allocate_TF_vec( nodes_rese );

    // vectors linked to t_nodes_rese
    t_position_xs = allocate_TF_vec( t_nodes_rese );
    t_position_ys = allocate_TF_vec( t_nodes_rese );
    t_normal_xs   = allocate_TF_vec( t_nodes_rese );
    t_normal_ys   = allocate_TF_vec( t_nodes_rese );
    t_cut_ids     = allocate_ST_vec( t_nodes_rese );
}

ConvexPolyhedron2::~ConvexPolyhedron2() {
    delete_TF_vec( position_xs  , nodes_rese   );
    delete_TF_vec( position_ys  , nodes_rese   );
    delete_TF_vec( normal_xs    , nodes_rese   );
    delete_TF_vec( normal_ys    , nodes_rese   );
    delete_ST_vec( cut_ids      , nodes_rese   );

    delete_TF_vec( distances    , nodes_rese   );

    delete_TF_vec( t_position_xs, t_nodes_rese );
    delete_TF_vec( t_position_ys, t_nodes_rese );
    delete_TF_vec( t_normal_xs  , t_nodes_rese );
    delete_TF_vec( t_normal_ys  , t_nodes_rese );
    delete_ST_vec( t_cut_ids    , t_nodes_rese );
}

void ConvexPolyhedron2::init_as_box( Pt pmin, Pt pmax, ST cut_id ) {
    // size (assuming that rese is enough)
    nodes_size = 4;

    // points
    position_xs[ 0 ] = pmin[ 0 ]; position_ys[ 0 ] = pmin[ 1 ];
    position_xs[ 1 ] = pmax[ 0 ]; position_ys[ 1 ] = pmin[ 1 ];
    position_xs[ 2 ] = pmax[ 0 ]; position_ys[ 2 ] = pmax[ 1 ];
    position_xs[ 3 ] = pmin[ 0 ]; position_ys[ 3 ] = pmax[ 1 ];

    // normals
    normal_xs[ 0 ] =  0; normal_ys[ 0 ] = -1;
    normal_xs[ 1 ] = +1; normal_ys[ 1 ] =  0;
    normal_xs[ 2 ] =  0; normal_ys[ 2 ] = +1;
    normal_xs[ 3 ] = -1; normal_ys[ 3 ] =  0;

    // cut_ids
    cut_ids[ 0 ] = cut_id;
    cut_ids[ 1 ] = cut_id;
    cut_ids[ 2 ] = cut_id;
    cut_ids[ 3 ] = cut_id;
}

void ConvexPolyhedron2::plane_cut( const FP64 **nds, const FP64 *nss, const U64 *nis, U64 nb_cuts ) {
    ST num_cut = 0;

    // if data can fit in registers
    #define METHOD_INCLUDE SDOT_STRINGIFY( SDOT_CONCAT_TOKEN_4_( internal/ConvexPolyhedron2_plane_cut_lt8, TF, ST, ARCH ) )
    #include METHOD_INCLUDE

    // if cut not finished
    for( ; ; ++num_cut ) {
        if ( num_cut == nb_cuts )
            return;

        TF nx = nds[ 0 ][ num_cut ];
        TF ny = nds[ 1 ][ num_cut ];
        TF ns = nss[ num_cut ];

        // get distance and outside bit for each node
        constexpr int ss = SimdSize<ARCH,TF>::value;
        if ( outside_nodes.size() <= ( nodes_size + 63 ) / 64 )
            outside_nodes.resize( ( nodes_size + 63 ) / 64 + 1 );
        for( ST i = 0; i < ( nodes_size + 63 ) / 64; ++i )
            outside_nodes[ i ] = 0;
        std::uint64_t ored_outside_nodes = 0;
        SimdRange<ss>::for_each( nodes_size, [&]( int n, auto s ) {
            using LF = SimdVec<TF,s.val>;

            LF px = LF::load_aligned( position_xs + n );
            LF py = LF::load_aligned( position_ys + n );
            LF bi = px * LF( nx ) + py * LF( ny );

            LF::store_aligned( distances + n, bi - LF( ns ) );
            std::uint64_t lo = bi > LF( ns ), sh = lo << n;
            outside_nodes[ n / 64 ] |= sh;
            ored_outside_nodes |= sh;
        } );

        // if nothing has changed => go to the next cut
        if ( ored_outside_nodes == 0 )
            continue;

        // we need room to store the new node data
        ST max_new_nodes_size = nodes_size + nodes_size / 2;
        if ( t_nodes_rese < max_new_nodes_size ) {
            while ( t_nodes_rese < max_new_nodes_size )
                t_nodes_rese *= 2;
            update_t_rese( t_nodes_rese );
        }

        // make a new edge set, in t storage
        auto outside_vec = [&]( int o ) -> bool {
            return outside_nodes[ o / 64 ] & ( 1ul << ( o % 64 ) );
        };
        int new_nodes_size = 0;
        for( ST n0 = 0, nm = nodes_size - 1; n0 < nodes_size; nm = n0++ ) {
            if ( outside_vec( n0 ) )
                continue;

            if ( outside_vec( nm ) ) {
                TF m = distances[ n0 ] / ( distances[ nm ] - distances[ n0 ] );
                t_position_xs[ new_nodes_size ] = position_xs[ n0 ] - m * ( position_xs[ nm ] - position_xs[ n0 ] );
                t_position_ys[ new_nodes_size ] = position_ys[ n0 ] - m * ( position_ys[ nm ] - position_ys[ n0 ] );
                t_cut_ids[ new_nodes_size ] = cut_ids[ nm ];
                ++new_nodes_size;
            }

            t_position_xs[ new_nodes_size ] = position_xs[ n0 ];
            t_position_ys[ new_nodes_size ] = position_ys[ n0 ];
            t_cut_ids[ new_nodes_size ] = cut_ids[ n0 ];
            ++new_nodes_size;

            int n1 = ( n0 + 1 ) % nodes_size;
            if ( outside_vec( n1 ) ) {
                TF m = distances[ n0 ] / ( distances[ n1 ] - distances[ n0 ] );
                t_position_xs[ new_nodes_size ] = position_xs[ n0 ] - m * ( position_xs[ n1 ] - position_xs[ n0 ] );
                t_position_ys[ new_nodes_size ] = position_ys[ n0 ] - m * ( position_ys[ n1 ] - position_ys[ n0 ] );
                t_cut_ids[ new_nodes_size ] = nis[ num_cut ];
                ++new_nodes_size;
            }
        }

        //
        if ( t_nodes_rese > nodes_rese ) {
            delete_TF_vec( distances, nodes_rese );
            distances = allocate_TF_vec( t_nodes_rese );
            outside_nodes.resize( ( t_nodes_rese + 63 ) / 64 );
        }

        std::swap( position_xs, t_position_xs );
        std::swap( position_ys, t_position_ys );
        std::swap( normal_xs, t_normal_xs );
        std::swap( normal_ys, t_normal_ys );
        std::swap( cut_ids, t_cut_ids );

        std::swap( nodes_rese, t_nodes_rese );
        nodes_size = new_nodes_size;
    }
}

void ConvexPolyhedron2::write_to_stream( std::ostream &os ) const {
    os << "pos: ";
    for( ST i = 0; i < nb_vertices(); ++i )
        os << ( i ? " [" : "[" ) << position_xs[ i ] << " " << position_ys[ i ] << "]";
}

void ConvexPolyhedron2::display_vtk( VtkOutput &vo ) const {
    std::vector<VtkOutput::Pt> points;
    for( ST i = 0; i < nb_vertices(); ++i )
        points.push_back( { position_xs[ i ], position_ys[ i ], TF( 0 ) } );
    points.push_back( { position_xs[ 0 ], position_ys[ 0 ], TF( 0 ) } );
    vo.add_lines( points );
}

void ConvexPolyhedron2::for_each_boundary( const std::function<void(const Boundary &)> &f ) {
    for( ST i = 0, j = nb_vertices() - 1; i < nb_vertices(); j = i++ )
        f( Boundary{ Edge<TF,2>{ { position_xs[ j ], position_ys[ j ] }, { position_xs[ i ], position_ys[ i ] } }, cut_ids[ j ] } );
}

TF* ConvexPolyhedron2::allocate_TF_vec( ST size ) {
    return new ( aligned_alloc( SimdAlig<ARCH,TF>::value * sizeof( TF ), size * sizeof( TF ) ) ) TF[ size ];
}

ST* ConvexPolyhedron2::allocate_ST_vec( ST size ) {
    return new ( aligned_alloc( SimdAlig<ARCH,ST>::value * sizeof( ST ), size * sizeof( ST ) ) ) ST[ size ];
}

void ConvexPolyhedron2::delete_TF_vec( TF *vec, ST size ) {
    for( ST i = size; i--; )
        vec[ i ].~TF();
    free( vec );
}

void ConvexPolyhedron2::delete_ST_vec( ST *vec, ST size ) {
    for( ST i = size; i--; )
        vec[ i ].~ST();
    free( vec );
}

void ConvexPolyhedron2::update_t_rese( ST t_rese ) {
    delete_TF_vec( t_position_xs, t_nodes_rese );
    delete_TF_vec( t_position_ys, t_nodes_rese );
    delete_TF_vec( t_normal_xs  , t_nodes_rese );
    delete_TF_vec( t_normal_ys  , t_nodes_rese );
    delete_ST_vec( t_cut_ids    , t_nodes_rese );

    t_position_xs = allocate_TF_vec( t_rese );
    t_position_ys = allocate_TF_vec( t_rese );
    t_normal_xs   = allocate_TF_vec( t_rese );
    t_normal_ys   = allocate_TF_vec( t_rese );
    t_cut_ids     = allocate_ST_vec( t_rese );

    t_nodes_rese = t_rese;
}

} // namespace sdot
#undef ConvexPolyhedron2

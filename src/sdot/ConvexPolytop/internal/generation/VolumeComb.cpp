#include "../../support/for_each_comb.h"
#include "../../support/ThreadPool.h"
#include "../../support/P.h"
#include "VolumeComb.h"
#include <algorithm>

VolumeComb::VolumeComb( std::vector<Pt> pts, std::vector<std::string> allowed_volume_types ) {
    // possible tetras
    if ( std::find( allowed_volume_types.begin(), allowed_volume_types.end(), "Tetra" ) != allowed_volume_types.end() ) {
        ::for_each_comb<TI>( pts.size(), 4, [&]( const TI *inds ) {
            Volume v;

            // nodes
            for( std::size_t i = 0; i < 4; ++i )
                v.nodes.push_back( inds[ i ] );

            // points
            for( std::size_t i = 0; i < 4; ++i )
                v.points.push_back( pts[ inds[ i ] ] );

            // volume
            v.volume = TetraAssembly::measure_tetra( v.points.data() );
            if ( v.volume == 0 )
                return;
            if ( v.volume < 0 ) {
                std::swap( v.points[ 0 ], v.points[ 1 ] );
                v.volume = - v.volume;
            }

            // tetra_assembly
            v.tetra_assembly.add_tetra( { v.points[ 0 ], v.points[ 1 ], v.points[ 2 ], v.points[ 3 ] } );

            possible_volumes.push_back( std::move( v ) );
        } );
    }

    // possible pyramids
    if ( std::find( allowed_volume_types.begin(), allowed_volume_types.end(), "Pyramid" ) != allowed_volume_types.end() ) {
        auto test_pyramid = [&]( std::array<TI,5> inds ) {
            Pt p0 = pts[ inds[ 0 ] ], p1 = pts[ inds[ 1 ] ], p2 = pts[ inds[ 2 ] ], p3 = pts[ inds[ 3 ] ], p4 = pts[ inds[ 4 ] ];

            // colinear edges
            if ( norm_2_p2( cross_prod( p1 - p0, p2 - p0 ) ) == 0 ) return;
            if ( norm_2_p2( cross_prod( p3 - p1, p0 - p1 ) ) == 0 ) return;
            if ( norm_2_p2( cross_prod( p2 - p3, p1 - p3 ) ) == 0 ) return;
            if ( norm_2_p2( cross_prod( p0 - p2, p3 - p2 ) ) == 0 ) return;

            // best {a,b} to get p3 = p0 + ( p1 - p0 ) * a + ( p2 - p0 ) * b
            TF maa = 0, mab = 0, mbb = 0, va = 0, vb = 0;
            for( TI d = 0; d < 3; ++d ) {
                maa += ( p1[ d ] - p0[ d ] ) * ( p1[ d ] - p0[ d ] );
                mab += ( p1[ d ] - p0[ d ] ) * ( p2[ d ] - p0[ d ] );
                mbb += ( p2[ d ] - p0[ d ] ) * ( p2[ d ] - p0[ d ] );
                va  += ( p1[ d ] - p0[ d ] ) * ( p3[ d ] - p0[ d ] );
                vb  += ( p2[ d ] - p0[ d ] ) * ( p3[ d ] - p0[ d ] );
            }

            TF det = maa * mbb - mab * mab;
            if ( det == 0 )
                return;

            TF a = ( va * mbb - vb * mab ) / det;
            TF b = ( maa * vb - mab * va ) / det;

            // not a planar face
            Pt g = p0 + a * ( p1 - p0 ) + b * ( p2 - p0 );
            if ( norm_2_p2( g - p3 ) != 0 )
                return;

            // cannot be converted to a convex face
            if ( a < 0 && b < 0 )
                return;

            // convert to a convex face if it's concave
            if ( a < 0 ) {
                std::swap( inds[ 2 ], inds[ 3 ] );
                std::swap( p2, p3 );
            }

            if ( b < 0 ) {
                std::swap( inds[ 1 ], inds[ 3 ] );
                std::swap( p1, p3 );
            }

            // flat volume
            Pt z = cross_prod( p1 - p0, p2 - p0 );
            if ( dot( p4 - p0, z ) == 0 )
                return;

            // wrong face orientation
            if ( dot( p4 - p0, z ) < 0 ) {
                std::swap( inds[ 1 ], inds[ 2 ] );
                std::swap( p1, p2 );
            }

            // nodes
            Volume v;
            for( std::size_t i = 0; i < inds.size(); ++i )
                v.nodes.push_back( inds[ i ] );

            // points
            for( std::size_t i = 0; i < inds.size(); ++i )
                v.points.push_back( pts[ v.nodes[ i ] ] );

            // volume
            Pt t0[] = { v.points[ 0 ], v.points[ 1 ], v.points[ 2 ], v.points[ 4 ] };
            Pt t1[] = { v.points[ 2 ], v.points[ 1 ], v.points[ 3 ], v.points[ 4 ] };
            v.volume = TetraAssembly::measure_tetra( t0 ) +
                       TetraAssembly::measure_tetra( t1 );

            // tetra_assembly
            v.tetra_assembly.add_pyramid( { v.points[ 0 ], v.points[ 1 ], v.points[ 2 ], v.points[ 3 ], v.points[ 4 ] } );

            // static int cpt = 0;
            // VtkOutput vo;
            // vo.add_pyramid( { v.points[ 0 ], v.points[ 1 ], v.points[ 2 ], v.points[ 3 ], v.points[ 4 ] } );
            // vo.save( "out_" + std::to_string( cpt++ ) + ".vtk" );
            // P( v.points[ 0 ], v.points[ 1 ], v.points[ 2 ], v.points[ 3 ], v.points[ 4 ], v.volume );

            possible_volumes.push_back( std::move( v ) );
        };
        ::for_each_comb<TI>( pts.size(), 5, [&]( const TI *inds ) {
            test_pyramid( { inds[ 1 ], inds[ 2 ], inds[ 3 ], inds[ 4 ], inds[ 0 ] } );
            test_pyramid( { inds[ 2 ], inds[ 3 ], inds[ 4 ], inds[ 0 ], inds[ 1 ] } );
            test_pyramid( { inds[ 3 ], inds[ 4 ], inds[ 0 ], inds[ 1 ], inds[ 2 ] } );
            test_pyramid( { inds[ 4 ], inds[ 0 ], inds[ 1 ], inds[ 2 ], inds[ 3 ] } );
            test_pyramid( { inds[ 0 ], inds[ 1 ], inds[ 2 ], inds[ 3 ], inds[ 4 ] } );
        } );
    }

    // are_disjoint vector
    are_disjoint.resize( possible_volumes.size() * possible_volumes.size() );
    for( TI n0 = 0, ind = 0; n0 < possible_volumes.size(); ++n0 ) {
        for( TI n1 = 0; n1 < possible_volumes.size(); ++n1, ++ind ) {
            TetraAssembly ta;
            ta.add_intersection( possible_volumes[ n0 ].tetra_assembly, possible_volumes[ n1 ].tetra_assembly );
            are_disjoint[ ind ] = ta.measure() == 0;
        }
    }
}

void VolumeComb::for_each_comb( const std::function<void(const std::vector<TI> &)> &f ) {
    std::atomic<TI> nb_done( 0 );
    TI nb_jobs = 16 * thread_pool.nb_threads();
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int /*num_thread*/ ) {
        TI beg = ( num_job + 0 ) * possible_volumes.size() / nb_jobs;
        TI end = ( num_job + 1 ) * possible_volumes.size() / nb_jobs;

        std::vector<bool> possible_set( possible_volumes.size(), true );
        for( TI ind = beg; ind < end; ++ind ) {
            for_each_comb_( f, possible_set, {}, ind );
            P( ++nb_done, possible_volumes.size() );
        }
    } );
}

void VolumeComb::for_each_comb_( const std::function<void(const std::vector<TI> &)> &f, std::vector<bool> possible_set, std::vector<TI> inds, TI ind ) {
    inds.push_back( ind );
    f( inds );

    // remove volumes that intersect the added volume
    for( TI j = ind + 1; j < possible_set.size(); ++j )
        possible_set[ j ] = possible_set[ j ] & are_disjoint[ ind * possible_set.size() + j ];

    // test with another element
    for( TI new_ind = ind + 1; new_ind < possible_set.size(); ++new_ind )
        if ( possible_set[ new_ind ] )
            for_each_comb_( f, possible_set, inds, new_ind );
}

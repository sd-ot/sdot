#include "TetraAssembly.h"

void TetraAssembly::add_intersection( const TetraAssembly &a, const TetraAssembly &b ) {
    for( const Tetra &tb : b.tetras ) {
        TetraAssembly tmp = a;
        tmp.plane_cut( tb.pts[ 2 ], cross_prod( tb.pts[ 0 ] - tb.pts[ 2 ], tb.pts[ 3 ] - tb.pts[ 2 ] ) );
        tmp.plane_cut( tb.pts[ 3 ], cross_prod( tb.pts[ 0 ] - tb.pts[ 3 ], tb.pts[ 1 ] - tb.pts[ 3 ] ) );
        tmp.plane_cut( tb.pts[ 3 ], cross_prod( tb.pts[ 1 ] - tb.pts[ 3 ], tb.pts[ 2 ] - tb.pts[ 3 ] ) );
        tmp.plane_cut( tb.pts[ 2 ], cross_prod( tb.pts[ 1 ] - tb.pts[ 2 ], tb.pts[ 0 ] - tb.pts[ 2 ] ) );
        for( const Tetra &rb : tmp.tetras )
            add_tetra( rb.pts );
    }
}

void TetraAssembly::add_pyramid( std::array<Pt,5> pts ) {
    tetras.push_back( { pts[ 0 ], pts[ 1 ], pts[ 2 ], pts[ 4 ] } );
    tetras.push_back( { pts[ 2 ], pts[ 1 ], pts[ 3 ], pts[ 4 ] } );
}

void TetraAssembly::add_tetra( std::array<Pt,4> pts ) {
    tetras.push_back( { pts } );
}

void TetraAssembly::add_wedge( std::array<Pt,6> pts ) {
    tetras.push_back( { pts[ 0 ], pts[ 1 ], pts[ 2 ], pts[ 3 ] } );
    tetras.push_back( { pts[ 3 ], pts[ 5 ], pts[ 4 ], pts[ 1 ] } );
    tetras.push_back( { pts[ 3 ], pts[ 2 ], pts[ 5 ], pts[ 1 ] } );
}

void TetraAssembly::add_hexa( std::array<Pt,8> pts ) {
    tetras.push_back( { pts[ 0 ], pts[ 4 ], pts[ 5 ], pts[ 6 ] } );
    tetras.push_back( { pts[ 3 ], pts[ 5 ], pts[ 7 ], pts[ 6 ] } );
    tetras.push_back( { pts[ 0 ], pts[ 6 ], pts[ 3 ], pts[ 2 ] } );
    tetras.push_back( { pts[ 0 ], pts[ 1 ], pts[ 3 ], pts[ 5 ] } );
    tetras.push_back( { pts[ 0 ], pts[ 6 ], pts[ 5 ], pts[ 3 ] } );
}

void TetraAssembly::plane_cut( Pt pos, Pt dir ) {
    std::vector<Tetra> new_tetras;

    for( const Tetra &tetra : tetras ) {
        TF x0 = tetra.pts[ 0 ][ 0 ], y0 = tetra.pts[ 0 ][ 1 ], z0 = tetra.pts[ 0 ][ 2 ];
        TF x1 = tetra.pts[ 1 ][ 0 ], y1 = tetra.pts[ 1 ][ 1 ], z1 = tetra.pts[ 1 ][ 2 ];
        TF x2 = tetra.pts[ 2 ][ 0 ], y2 = tetra.pts[ 2 ][ 1 ], z2 = tetra.pts[ 2 ][ 2 ];
        TF x3 = tetra.pts[ 3 ][ 0 ], y3 = tetra.pts[ 3 ][ 1 ], z3 = tetra.pts[ 3 ][ 2 ];

        TF s0 = ( x0 - pos[ 0 ] ) * dir[ 0 ] + ( y0 - pos[ 1 ] ) * dir[ 1 ] + ( z0 - pos[ 2 ] ) * dir[ 2 ];
        TF s1 = ( x1 - pos[ 0 ] ) * dir[ 0 ] + ( y1 - pos[ 1 ] ) * dir[ 1 ] + ( z1 - pos[ 2 ] ) * dir[ 2 ];
        TF s2 = ( x2 - pos[ 0 ] ) * dir[ 0 ] + ( y2 - pos[ 1 ] ) * dir[ 1 ] + ( z2 - pos[ 2 ] ) * dir[ 2 ];
        TF s3 = ( x3 - pos[ 0 ] ) * dir[ 0 ] + ( y3 - pos[ 1 ] ) * dir[ 1 ] + ( z3 - pos[ 2 ] ) * dir[ 2 ];

        bool o0 = s0 > 0;
        bool o1 = s1 > 0;
        bool o2 = s2 > 0;
        bool o3 = s3 > 0;

        #define ADD_TETRA( a, b, c, d ) \
            new_tetras.push_back( { Pt{ x##a, y##a, z##a }, Pt{ x##b, y##b,  z##b }, Pt{ x##c, y##c, z##c }, Pt{ x##d, y##d,  z##d } } )

        auto cut_1_outside = [&]( TF x0, TF y0, TF z0, TF x1, TF y1, TF z1, TF x2, TF y2, TF z2, TF x3, TF y3, TF z3, TF s4, TF s5, TF s6 ) {
            TF x4 = x0 + s4 * ( x1 - x0 );
            TF x5 = x0 + s5 * ( x2 - x0 );
            TF x6 = x0 + s6 * ( x3 - x0 );

            TF y4 = y0 + s4 * ( y1 - y0 );
            TF y5 = y0 + s5 * ( y2 - y0 );
            TF y6 = y0 + s6 * ( y3 - y0 );

            TF z4 = z0 + s4 * ( z1 - z0 );
            TF z5 = z0 + s5 * ( z2 - z0 );
            TF z6 = z0 + s6 * ( z3 - z0 );

            ADD_TETRA( 1, 2, 5, 3 );
            ADD_TETRA( 1, 5, 4, 3 );
            ADD_TETRA( 4, 6, 3, 5 );
        };


        auto cut_2_outside = [&]( TF x0, TF y0, TF z0, TF x1, TF y1, TF z1, TF x2, TF y2, TF z2, TF x3, TF y3, TF z3, TF s02, TF s03, TF s12, TF s13 ) {
            TF x4 = x0 + s02 * ( x2 - x0 );
            TF x5 = x0 + s03 * ( x3 - x0 );
            TF x6 = x1 + s12 * ( x2 - x1 );
            TF x7 = x1 + s13 * ( x3 - x1 );

            TF y4 = y0 + s02 * ( y2 - y0 );
            TF y5 = y0 + s03 * ( y3 - y0 );
            TF y6 = y1 + s12 * ( y2 - y1 );
            TF y7 = y1 + s13 * ( y3 - y1 );

            TF z4 = z0 + s02 * ( z2 - z0 );
            TF z5 = z0 + s03 * ( z3 - z0 );
            TF z6 = z1 + s12 * ( z2 - z1 );
            TF z7 = z1 + s13 * ( z3 - z1 );

            ADD_TETRA( 2, 4, 6, 5 );
            ADD_TETRA( 2, 3, 5, 6 );
            ADD_TETRA( 3, 7, 5, 6 );
        };


        auto cut_3_outside = [&]( TF x0, TF y0, TF z0, TF x1, TF y1, TF z1, TF x2, TF y2, TF z2, TF x3, TF y3, TF z3, TF s0, TF s1, TF s2 ) {
            TF x4 = x3 + s0 * ( x0 - x3 );
            TF x5 = x3 + s1 * ( x1 - x3 );
            TF x6 = x3 + s2 * ( x2 - x3 );

            TF y4 = y3 + s0 * ( y0 - y3 );
            TF y5 = y3 + s1 * ( y1 - y3 );
            TF y6 = y3 + s2 * ( y2 - y3 );

            TF z4 = z3 + s0 * ( z0 - z3 );
            TF z5 = z3 + s1 * ( z1 - z3 );
            TF z6 = z3 + s2 * ( z2 - z3 );

            ADD_TETRA( 4, 5, 6, 3 );
        };

        #define C1( A, B, C, D ) cut_1_outside( x##A, y##A, z##A, x##B, y##B, z##B, x##C, y##C, z##C, x##D, y##D, z##D, s##A / ( s##A - s##B ), s##A / ( s##A - s##C ), s##A / ( s##A - s##D ) )
        #define C2( A, B, C, D ) cut_2_outside( x##A, y##A, z##A, x##B, y##B, z##B, x##C, y##C, z##C, x##D, y##D, z##D, s##A / ( s##A - s##C ), s##A / ( s##A - s##D ), s##B / ( s##B - s##C ), s##B / ( s##B - s##D ) )
        #define C3( A, B, C, D ) cut_3_outside( x##A, y##A, z##A, x##B, y##B, z##B, x##C, y##C, z##C, x##D, y##D, z##D, s##D / ( s##D - s##A ), s##D / ( s##D - s##B ), s##D / ( s##D - s##C ) )

        switch ( 1 * o0 + 2 * o1 + 4 * o2 + 8 * o3 ) {
        case 1 * 0 + 2 * 0 + 4 * 0 + 8 * 0: // all inside
            new_tetras.push_back( tetra );
            break;
        case 1 * 1 + 2 * 0 + 4 * 0 + 8 * 0: // o0
            C1( 0, 1, 2, 3 );
            break;
        case 1 * 0 + 2 * 1 + 4 * 0 + 8 * 0: // o1
            C1( 1, 2, 0, 3 );
            break;
        case 1 * 1 + 2 * 1 + 4 * 0 + 8 * 0: // o0, o1
            C2( 0, 1, 2, 3 );
            break;
        case 1 * 0 + 2 * 0 + 4 * 1 + 8 * 0: // o2
            C1( 2, 0, 1, 3 );
            break;
        case 1 * 1 + 2 * 0 + 4 * 1 + 8 * 0: // o0, o2
            C2( 0, 2, 3, 1 );
            break;
        case 1 * 0 + 2 * 1 + 4 * 1 + 8 * 0: // o1, o2
            C2( 1, 2, 0, 3 );
            break;
        case 1 * 1 + 2 * 1 + 4 * 1 + 8 * 0: // o0, o1, o2
            C3( 0, 1, 2, 3 );
            break;
        case 1 * 0 + 2 * 0 + 4 * 0 + 8 * 1: // o3
            C1( 3, 0, 2, 1 );
            break;
        case 1 * 1 + 2 * 0 + 4 * 0 + 8 * 1: // o0, o3
            C2( 0, 3, 1, 2 );
            break;
        case 1 * 0 + 2 * 1 + 4 * 0 + 8 * 1: // o1, o3
            C2( 1, 3, 2, 0 );
            break;
        case 1 * 1 + 2 * 1 + 4 * 0 + 8 * 1: // o0, o1, o3
            C3( 0, 3, 1, 2 );
            break;
        case 1 * 0 + 2 * 0 + 4 * 1 + 8 * 1: // o2, o3
            C2( 2, 3, 0, 1 );
            break;
        case 1 * 1 + 2 * 0 + 4 * 1 + 8 * 1: // o0, o2, o3
            C3( 0, 2, 3, 1 );
            break;
        case 1 * 0 + 2 * 1 + 4 * 1 + 8 * 1: // o1, o2, o3
            C3( 1, 3, 2, 0 );
            break;
        case 1 * 1 + 2 * 1 + 4 * 1 + 8 * 1: // o0, o1, o2, o3
            // all outside
            break;
        }
    }

    std::swap( tetras, new_tetras );
}

TetraAssembly::TF TetraAssembly::measure() const {
    TF res = 0;
    for( const Tetra &t : tetras )
        res += measure_tetra( t.pts.data() );
    return res;
}

TetraAssembly::TF TetraAssembly::measure_tetra( const Pt *pts ) {
    TF x0 = pts[ 0 ][ 0 ];
    TF y0 = pts[ 0 ][ 1 ];
    TF z0 = pts[ 0 ][ 2 ];
    TF x1 = pts[ 1 ][ 0 ] - x0;
    TF y1 = pts[ 1 ][ 1 ] - y0;
    TF z1 = pts[ 1 ][ 2 ] - z0;
    TF x2 = pts[ 2 ][ 0 ] - x0;
    TF y2 = pts[ 2 ][ 1 ] - y0;
    TF z2 = pts[ 2 ][ 2 ] - z0;
    TF x3 = pts[ 3 ][ 0 ] - x0;
    TF y3 = pts[ 3 ][ 1 ] - y0;
    TF z3 = pts[ 3 ][ 2 ] - z0;

    return TF( 1 ) / 6 * (
        x1 * ( y2 * z3 - y3 * z2 ) -
        y1 * ( x2 * z3 - x3 * z2 ) +
        z1 * ( x2 * y3 - x3 * y2 )
    );
}

void TetraAssembly::display_vtk( VtkOutput &vo ) const {
    for( const Tetra &t : tetras ) {
        std::array<VtkOutput::Pt,4> pts;
        for( TI i = 0; i < pts.size(); ++i )
            pts[ i ] = t.pts[ i ];
        vo.add_tetra( pts );
    }
}

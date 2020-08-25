#include "../support/for_each_comb.h"
#include "../support/ASSERT.h"
#include "../support/TODO.h"
#include "../support/P.h"
#include <cmath>

#include "SimplexSumPolytop.h"

template<int dim,class TF,class TI,class Arch>
SimplexSumPolytop<dim,TF,TI,Arch>::SimplexSumPolytop( std::array<Pt,dim+1> points ) : positions( points.begin(), points.end() ), nb_cuts( 1 ) {
    Simplex simplex;
    for( TI d = 0; d < dim + 1; ++d ) {
        simplex.cut_ids[ d ] = 0;
        simplex.nodes[ d ] = d;
    }
    if ( measure_( simplex ) < 0 )
        std::swap( simplex.nodes[ 0 ], simplex.nodes[ 1 ] );

    simplices.push_back( simplex );
}

template<int dim,class TF,class TI,class Arch>
SimplexSumPolytop<dim,TF,TI,Arch>::SimplexSumPolytop( Pt center, TF radius ) : nb_cuts( 1 ) {
    TF s = 100000, l = radius * s / TF( s * std::sqrt( dim * std::pow( 1.0 / dim - 1.0 / ( dim + 1 ), 2 ) ) - 1 );
    Pt B = center - l / ( dim + 1 );

    positions.resize( dim + 1 );
    positions[ 0 ] = B;
    for( TI d = 0; d < dim; ++d ) {
        positions[ d + 1 ] = B;
        positions[ d + 1 ][ d ] += l;
    }

    Simplex simplex;
    for( TI d = 0; d < dim + 1; ++d ) {
        simplex.cut_ids[ d ] = 0;
        simplex.nodes[ d ] = d;
    }
    simplices.push_back( simplex );
}

template<int dim,class TF,class TI,class Arch>
void SimplexSumPolytop<dim,TF,TI,Arch>::write_to_stream( std::ostream &os ) const {
    for( const Simplex &s : simplices ) {
        for( TI d = 0; d < dim + 1; ++d )
            os << positions[ s.nodes[ d ] ] << " ";
        os << "\n";
    }
}

template<int dim,class TF,class TI,class Arch>
void SimplexSumPolytop<dim,TF,TI,Arch>::display_vtk( VtkOutput &vo ) const {
    for( const Simplex &simplex : simplices ) {
        std::array<TI,3> triangle;
        for_each_comb<TI>( 3, dim + 1, triangle.data(), [&](TI *) {
            std::array<VtkOutput::Pt,3> pts;
            for( TI i = 0; i < 3; ++i )
                for( TI d = 0; d < dim; ++d )
                    pts[ i ][ d ] = conv( positions[ simplex.nodes[ triangle[ i ] ] ][ d ], S<typename VtkOutput::TF>() );
            vo.add_triangle( pts );
        } );
    }
}

template<int dim,class TF,class TI,class Arch>
TI SimplexSumPolytop<dim,TF,TI,Arch>::plane_cut( Pt pos, Pt dir ) {
    // scalar products
    std::vector<TF> sps( positions.size() );
    for( TI i = 0; i < sps.size(); ++i )
        sps[ i ] = dot( positions[ i ] - pos, dir );

    std::vector<TI> new_nodes( positions.size() * ( positions.size() - 1 ) / 2, 0 ); // for each possible edge
    std::vector<Simplex> new_simplices;
    for( const Simplex &simplex : simplices ) {
        Simplex new_simplex;
        std::vector<GenSimplex<dim-1>> closers;
        std::vector<bool> allowed_nodes( positions.size(), true );
        plane_cut_( new_simplices, new_simplex, allowed_nodes, new_nodes.data(), sps, simplex, closers );
    }
    simplices = std::move( new_simplices );
    return nb_cuts++;
}

template<int dim,class TF,class TI,class Arch> template<int nvi>
void SimplexSumPolytop<dim,TF,TI,Arch>::plane_cut_( std::vector<Simplex> &new_simplices, Simplex &new_simplex, std::vector<bool> &allowed_nodes, TI *new_nodes, const std::vector<TF> &sps, const GenSimplex<nvi> &simplex, std::vector<GenSimplex<nvi-1>> &closers ) {
    // find a base point (allowed, and inside the cut)
    // -> il faut continuer qd même pour avoir les closers
    TI base_point;
    for( TI i = 0; ; ++i ) {
        if ( i == nvi + 1 )
            return;
        if ( ! ( sps[ simplex.nodes[ i ] ] > 0 ) ) {
            new_simplex.nodes[ dim - nvi ] = i;
            allowed_nodes[ i ] = false;
            base_point = i;
            break;
        }
    }

    // rec call for each sub_item (i.e. for each edge if triangle, for each triangle if tetra, ...)
    std::vector<GenSimplex<nvi-2>> sub_closers;
    for( TI num_face = 0; num_face < nvi + 1; ++num_face ) {
        GenSimplex<nvi-1> sub_item;
        for( TI n = 0; n < nvi; ++n )
            sub_item.nodes[ n ] = simplex.nodes[ n + ( n >= num_face ) ];
        plane_cut_( new_simplices, new_simplex, allowed_nodes, new_nodes, sps, sub_item, sub_closers );
    }

    P( sub_closers.size() );

    //        std::vector<GenSimplex<nvi-1>> new_loc_faces;
    //        plane_cut_( new_loc_faces, new_sub_faces, new_nodes, pos, dir, face, loc_first_inside_node );
    //        if ( nvi == 3 )
    //            P( new_loc_faces.size() );

    //        for( const GenSimplex<nvi-1> &new_face : new_loc_faces ) {
    //            bool valid = true;
    //            GenSimplex<nvi> new_simplex;
    //            for( TI n = 0; n < nvi; ++n ) {
    //                valid &= new_face.nodes[ n ] != loc_first_inside_node;
    //                new_simplex.nodes[ n ] = new_face.nodes[ n ];
    //            }

    //            if ( valid ) {
    //                new_simplex.nodes[ nvi ] = loc_first_inside_node;
    //                new_simplices.push_back( new_simplex );
    //            }
    //        }
    //    }

    //    if ( nvi == 3 ) {
    //        P( new_sub_faces.size() );
    //        for( const GenSimplex<nvi-2> new_sub_face : new_sub_faces )
    //            P( new_sub_face.nodes);
    //        return;
    //    }

    //    // close
    //    for( const GenSimplex<nvi-2> new_sub_face : new_sub_faces ) {
    //        bool valid = true;
    //        GenSimplex<nvi-1> new_face;
    //        for( TI n = 0; n < nvi - 1; ++n ) {
    //            valid &= new_sub_face.nodes[ n ] != new_sub_faces[ 0 ].nodes[ 0 ];
    //            new_face.nodes[ n ] = new_sub_face.nodes[ n ];
    //        }

    //        if ( valid ) {
    //            GenSimplex<nvi> new_simplex;
    //            for( TI n = 0; n < nvi - 1; ++n )
    //                new_simplex.nodes[ n ] = new_face.nodes[ n ];
    //            new_simplex.nodes[ nvi - 1 ] = new_sub_faces[ 0 ].nodes[ 0 ];
    //            new_simplex.nodes[ nvi ] = loc_first_inside_node;
    //            new_simplices.push_back( new_simplex );

    //            new_face.nodes[ nvi - 1 ] = new_sub_faces[ 0 ].nodes[ 0 ];
    //            if ( nvi < dim )
    //                new_faces.push_back( new_face );
    //        }
    //    }

    //    //
    //    if ( first_inside_node == std::numeric_limits<TI>::max() )
    //        first_inside_node = loc_first_inside_node;
}

template<int dim,class TF,class TI,class Arch>
void SimplexSumPolytop<dim,TF,TI,Arch>::plane_cut_( std::vector<Simplex> &new_simplices, Simplex &new_simplex, std::vector<bool> &allowed_nodes, TI *new_nodes, const std::vector<TF> &sps, const GenSimplex<1> &simplex, std::vector<GenSimplex<0>> &closers ) {
    TI n0 = simplex.nodes[ 0 ];
    TI n1 = simplex.nodes[ 1 ];
    TF s0 = sps[ n0 ];
    TF s1 = sps[ n1 ];
    bool o0 = s0 > 0;
    bool o1 = s1 > 0;

    if ( o0 && o1 )
        return;

    auto new_node = [&]() {
        TI o0 = std::min( n0, n1 );
        TI o1 = std::max( n0, n1 );
        TI nn = o1 * ( o1 - 1 ) / 2 + o0;
        if ( ! new_nodes[ nn ] ) {
            new_nodes[ nn ] = positions.size();
            positions.push_back( positions[ n0 ] + s0 / ( s0 - s1 ) * ( positions[ n1 ] - positions[ n0 ] ) );
        }
        closers.push_back( { .nodes = { new_nodes[ nn ] } } );
        return new_nodes[ nn ];
    };

    auto push = [&]( TI n0, TI n1 ) {
        new_simplex.nodes[ dim - 1 ] = n0;
        new_simplex.nodes[ dim - 0 ] = n1;
        new_simplices.push_back( new_simplex );
    };

    // only n1 is inside
    if ( o0 ) {
        if ( allowed_nodes[ n1 ] )
            push( new_node(), n1 );
        return;
    }

    // only n0 is inside
    if ( o1 ) {
        if ( allowed_nodes[ n0 ] )
            push( n0, new_node() );
        return;
    }

    // n0 and n1 are both inside
    if ( allowed_nodes[ n0 ] && allowed_nodes[ n1 ] )
        push( n0, n1 );
}

template<int dim,class TF,class TI,class Arch>
TF SimplexSumPolytop<dim,TF,TI,Arch>::measure_( const Simplex &simplex ) const {
    std::array<Pt,dim> dirs;
    for( TI d = 0; d < dim; ++d )
        dirs[ d ] = positions[ simplex.nodes[ d + 1 ] ] - positions[ simplex.nodes[ 0 ] ];
    return determinant( dirs[ 0 ].data, N<dim>() );
}

template<int dim,class TF,class TI,class Arch>
TF SimplexSumPolytop<dim,TF,TI,Arch>::measure() const {
    TF res = 0;
    for( const Simplex &simplex : simplices )
        res += measure_( simplex );
    return res / factorial( TF( dim ) );
}

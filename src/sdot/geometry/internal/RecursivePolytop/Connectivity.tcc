#include "../../../support/for_each_comb.h"
#include "../../../support/ASSERT.h"
#include "../../../support/P.h"
#include "ConnectivityPool.h"

namespace sdot {
namespace internal {
namespace RecursivePolytop {

// write_to_stream -----------------------------------------------------------------------
template<class TI,int nvi>
void Connectivity<TI,nvi>::write_to_stream( std::ostream &os ) const {
    os << "[";
    for( TI i = 0; i < boundaries.size(); ++i )
        boundaries[ i ]->write_to_stream( os << ( i++ ? "," : "" ) );
    os << "]";
}

template<class TI>
void Connectivity<TI,0>::write_to_stream( std::ostream &os ) const {
    os << node_number;
}

// New ------------------------------------------------------------------------------------
template<class TI,int nvi> template<class Pt>
void Connectivity<TI,nvi>::add_convex_hull( std::vector<Ocn> &res, Cpl &item_pool, BumpPointerPool &mem_pool, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &center ) {
    using BoundaryConnectivity = Connectivity<TI,nvi-1>;
    static constexpr int dim = Pt::dim;
    using TF = typename Pt::TF;

    std::vector<Obn> oriented_boundaries;
    std::vector<Pt> boundary_normals;

    // try each possible vertex selection to make new faces
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal
        Pt orig = positions[ indices[ chosen_num_indices[ 0 ] ] ];
        for( TI d = 0; d < nvi - 1; ++d )
            normals[ dim - nvi + d ] = positions[ indices[ chosen_num_indices[ d + 1 ] ] ] - orig;
        Pt face_normal = cross_prod( normals );
        normals[ dim - nvi ] = face_normal;

        // test if we already have this face
        for( TI i = 0; i < oriented_boundaries.size(); ++i )
            if ( dot( boundary_normals[ i ], orig - positions[ oriented_boundaries[ i ].connectivity->first_vertex()->node_number ] ) == 0 && colinear( boundary_normals[ i ], face_normal ) )
                return;

        // test in and out points
        bool has_ins = false;
        bool has_out = false;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice ) {
            TF d = dot( positions[ indices[ num_indice ] ] - orig, face_normal );
            has_ins |= d < 0;
            has_out |= d > 0;
        }

        if ( has_ins && has_out )
            return;

        // update normal orientation if necessary
        if ( has_out )
            face_normal *= TF( -1 );

        // find all the points that belong to this face
        TI *new_indices = indices + nb_indices + nvi, new_nb_indices = 0;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
            if ( dot( positions[ indices[ num_indice ] ] - orig, face_normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // center
        Pt face_center = TF( 0 );
        for( TI i = 0; i < new_nb_indices; ++i )
            face_center += positions[ new_indices[ i ] ];
        face_center /= TF( new_nb_indices );

        // update of prev_dirs
        dirs[ dim - nvi ] = face_center - center;

        // add the new face
        BoundaryConnectivity::add_convex_hull( oriented_boundaries, item_pool.next, mem_pool, positions, new_indices, new_nb_indices, normals, dirs, face_center );
        boundary_normals.push_back( face_normal );
    } );

    // create a new item
    if ( oriented_boundaries.size() ) {
        std::sort( oriented_boundaries.begin(), oriented_boundaries.end() );
        res.push_back( item_pool.find_or_create( mem_pool, std::move( oriented_boundaries ) ) );
    }
}

template<class TI> template<class Pt>
void Connectivity<TI,0>::add_convex_hull( std::vector<Ocn> &res, Cpl &item_pool, BumpPointerPool &mem_pool, const Pt */*positions*/, TI *indices, TI /*nb_indices*/, Pt */*normals*/, Pt *dirs, const Pt &/*center*/ ) {
    bool neg = determinant( dirs->data, N<Pt::dim>() ) < 0;
    res.push_back( item_pool.find_or_create( mem_pool, *indices, neg ) );
}

// copy -----------------------------------------------------------------------
template<class TI,int nvi> template<class Pt>
Connectivity<TI,nvi> *Connectivity<TI,nvi>::copy_rec( std::vector<Pt> &new_positions, Cpl &new_item_pool, BumpPointerPool &new_mem_pool, const std::vector<Pt> &old_positions ) const {
    if ( ! new_item ) {
        std::vector<Obn> new_boundaries( boundaries.size() );
        for( TI i = 0; i < boundaries.size(); ++i )
            new_boundaries[ i ] = { boundaries[ i ].connectivity->copy_rec( new_positions, new_item_pool.next, new_mem_pool, old_positions ), boundaries[ i ].neg };
        std::sort( new_boundaries.begin(), new_boundaries.end() );

        new_item = new_item_pool.create( new_mem_pool, std::move( new_boundaries ) );
    }

    return new_item;
}

template<class TI> template<class Pt>
Connectivity<TI,0> *Connectivity<TI,0>::copy_rec( std::vector<Pt> &new_positions, Cpl &new_item_pool, BumpPointerPool &new_mem_pool, const std::vector<Pt> &old_positions ) const {
    if ( ! new_item ) {
        new_item = new_item_pool.create( new_mem_pool, new_positions.size() );
        new_positions.push_back( old_positions[ node_number ] );
    }

    return new_item;
}

// for_each_possibility -------------------------------------------------------
template<class TI,int nvi>
void Connectivity<TI,nvi>::for_each_possibility( const std::function<void( const std::vector<std::vector<Obn>> &proposition )> &f, std::vector<std::vector<Obn>> &proposition, TI n ) const {
    if ( n == boundaries.size() ) {
        f( proposition );
        return;
    }

    for( TI i = 0; i < boundaries[ n ].connectivity->new_items.size(); ++i ) {
        TI nb_items = boundaries[ n ].connectivity->new_items.size();
        proposition[ n ].resize( nb_items );
        for( TI i = 0; i < nb_items; ++i )
            proposition[ n ][ i ] = { boundaries[ n ].connectivity->new_items[ n ][ i ], boundaries[ n ].neg }; // hum
        for_each_possibility( f, proposition, n + 1 );
    }
}

template<class TI,int nvi>
void Connectivity<TI,nvi>::for_each_possibility( const std::function<void( const std::vector<std::vector<Obn>> &proposition )> &f ) const {
    std::vector<std::vector<Obn>> proposition( new_items.size() );
    for_each_possibility( f, proposition, 0 );
}

// copy -----------------------------------------------------------------------
template<class TI,int nvi> template<int n>
void Connectivity<TI,nvi>::conn_cut( Cpl &/*new_item_pool*/, Mpl &/*new_mem_pool*/, N<n>, const std::function<TI(TI,TI)> &/*interp*/ ) const {
    TODO;
}

template<class TI,int nvi>
void Connectivity<TI,nvi>::conn_cut( Cpl &new_item_pool, Mpl &new_mem_pool, N<2>, const std::function<TI(TI,TI)> &/*interp*/ ) const {
    new_items.clear();
    for_each_possibility( [&]( const std::vector<std::vector<Obn>> &proposition ) {
        std::vector<Obn> new_edges;
        for( const std::vector<Obn> &p : proposition ) {
            if ( p.empty() )
                continue;
            if ( p.size() != 1 )
                TODO; // several volumes
            new_edges.push_back( p[ 0 ] );
        }
        // new possibility (with only one volume)
        new_items.push_back( { new_item_pool.create( new_mem_pool, std::move( new_edges ) ) } );
    } );

}

template<class TI,int nvi>
void Connectivity<TI,nvi>::conn_cut( Cpl &new_item_pool, Mpl &new_mem_pool, N<1>, const std::function<TI(TI,TI)> &interp ) const {
    ASSERT( boundaries.size() == 2 );
    TI s = boundaries[ 0 ].neg;

    Vtx *v0 = boundaries[ 1 - s ].connectivity;
    Vtx *v1 = boundaries[ 0 + s ].connectivity;

    bool o0 = v0->new_items[ 0 ].empty();
    bool o1 = v1->new_items[ 0 ].empty();

    // helper
    auto new_node = [&]() {
        return new_item_pool.next.create( new_mem_pool, interp( v0->node_number, v1->node_number ) );
    };

    // all outside
    if ( o0 && o1 ) {
        new_items = { {} };
        return;
    }

    // only v0 is outside
    if ( o0 ) {
        new_items = { { new_item_pool.create( new_mem_pool, { { new_node(), true }, { v1->new_items[ 0 ][ 0 ], false } } ) } };
        return;
    }

    // only v1 is outside
    if ( o1 ) {
        new_items = { { new_item_pool.create( new_mem_pool, { { v0->new_items[ 0 ][ 0 ], true }, { new_node(), false } } ) } };
        return;
    }

    // all inside
    new_items = { { new_item_pool.create( new_mem_pool, { { v0->new_items[ 0 ][ 0 ], true }, { v1->new_items[ 0 ][ 0 ], false } } ) } };
}

template<class TI>
void Connectivity<TI,0>::conn_cut( Cpl &/*new_item_pool*/, Mpl &/*new_mem_pool*/, N<0>, const std::function<TI(TI,TI)> &/*interp*/ ) const {
    // nothing to do
}


} // namespace sdot
} // namespace internal
} // namespace RecursivePolytop

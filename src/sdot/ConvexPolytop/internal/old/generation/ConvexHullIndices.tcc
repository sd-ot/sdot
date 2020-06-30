#include "../../../support/for_each_comb.h"
#include "../../../support/TODO.h"
#include "../../../support/range.h"
#include "../../../support/P.h"
#include "ConvexHullIndices.h"
#include <iterator>

template<class TF> template<class Pu>
ConvexHullIndices<1,TF>::ConvexHullIndices( const std::vector<Pt> &pts, const std::vector<TI> &inds, const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs ) : inds{ inds } {
    if ( inds.size() < 2 )
        return;

    // keep only a min and a max point
    TI mi = 0, ma = 0;
    for( TI i = 1; i < pts.size(); ++i ) {
        if ( pts[ mi ][ 0 ] > pts[ i ][ 0 ] )
            mi = i;
        if ( pts[ ma ][ 0 ] < pts[ i ][ 0 ] )
            ma = i;
    }
    this->inds = {
        inds[ mi ],
        inds[ ma ]
    };

    // check orientation
    adjust_orientation( ori_pts, ori_dirs );
}

template<class TF>
ConvexHullIndices<1,TF>::ConvexHullIndices( const std::vector<Pt> &pts ) : ConvexHullIndices( pts, range<TI>( pts.size() ), pts, {} ) {
}

template<class TF> template<class Pu>
void ConvexHullIndices<1,TF>::adjust_orientation( const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs ) {
    if ( measure( ori_pts, ori_dirs ) < 0 )
        std::swap( inds[ 0 ], inds[ 1 ] );
}

template<class TF>
void ConvexHullIndices<1,TF>::get_links_rec( std::vector<std::set<TI>> &links ) const {
    TI m = std::max( inds[ 0 ], inds[ 1 ] );
    if ( links.size() <= m )
        links.resize( m + 1 );
    links[ inds[ 0 ] ].insert( inds[ 1 ] );
    links[ inds[ 1 ] ].insert( inds[ 0 ] );
}

template<class TF>
bool ConvexHullIndices<1,TF>::operator<( const ConvexHullIndices &that ) const {
    return inds < that.inds;
}

template<class TF>
void ConvexHullIndices<1,TF>::write_to_stream( std::ostream &os, std::string sp ) const {
    os << sp << inds;
}

template<class TF> template<class Pu>
TF ConvexHullIndices<1,TF>::measure( const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs, TF div ) const {
    std::vector<Pu> new_ori_dirs = ori_dirs;
    new_ori_dirs.push_back( ori_pts[ inds.back() ] - ori_pts[ inds.front() ] );
    return determinant( &new_ori_dirs[ 0 ][ 0 ], N<Pu::dim>() ) / div;
}

template<class TF> template<class Pu>
Pu ConvexHullIndices<1,TF>::center( const std::vector<Pu> &ori_pts ) const {
    return ( ori_pts[ inds[ 1 ] ] + ori_pts[ inds[ 0 ] ] ) / TF( 2 );
}

// ----------------------------------------------------------------------------------------------------------------

template<int dim,class TF> template<class Pu>
ConvexHullIndices<dim,TF>::ConvexHullIndices( const std::vector<Pt> &loc_pts, const std::vector<TI> &loc_inds, const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs ) {
    struct Face { std::array<Pt,dim-1> dirs; Pt normal, orig; };
    std::vector<Face> faces;

    // from all the points...
    for( TI pivot = 0; pivot < loc_pts.size(); ++pivot ) {
        // for all the possible n - 1 other points...
        for_each_comb<TI>( loc_pts.size() - 1, dim - 1, [&]( TI *shifted_ind_other_points ) {
            // make a face (to store it if it's new)
            Face face;
            face.orig = loc_pts[ pivot ];
            for( TI d = 0; d < dim - 1; ++d )
                face.dirs[ d ]= loc_pts[ shifted_ind_other_points[ d ] + ( shifted_ind_other_points[ d ] >= pivot ) ] - loc_pts[ pivot ];

            // get a normal
            face.normal = cross_prod( face.dirs.data() );
            if ( norm_2_p2( face.normal ) == TF( 0 ) )
                return;

            // stop here if we already have this face
            for( const Face &ecaf : faces )
                if ( dot( ecaf.normal, loc_pts[ pivot ] - ecaf.orig ) == 0 && colinear( ecaf.normal, face.normal ) )
                    return;

            // stop here if it's not an "exterior" face
            bool has_ins = false;
            bool has_out = false;
            for( TI op = 0; op < loc_pts.size(); ++op ) {
                if ( op != pivot ) {
                    TF d = dot( loc_pts[ op ] - loc_pts[ pivot ], face.normal );
                    has_ins |= d < 0;
                    has_out |= d > 0;
                }
            }
            if ( has_ins && has_out )
                return;

            // update normal orientation if necessary
            if ( has_out )
                face.normal *= TF( -1 );

            // find all the points that belong to this face
            std::vector<Point<TF,dim-1>> new_pts;
            std::vector<TI> new_inds;
            for( TI op = 0; op < loc_pts.size(); ++op ) {
                if ( dot( loc_pts[ op ] - loc_pts[ pivot ], face.normal ) == TF( 0 ) ) {
                    // projection
                    std::array<std::array<TF,dim-1>,dim-1> M;
                    std::array<TF,dim-1> V;
                    for( TI r = 0; r < dim - 1; ++r )
                        for( TI c = 0; c < dim - 1; ++c )
                            M[ r ][ c ] = 0;
                    for( TI r = 0; r < dim - 1; ++r )
                        V[ r ] = 0;
                    for( TI d = 0; d < dim; ++d ) {
                        for( TI r = 0; r < dim - 1; ++r )
                            for( TI c = 0; c < dim - 1; ++c )
                                M[ r ][ c ] += face.dirs[ r ][ d ] * face.dirs[ c ][ d ];
                        for( TI r = 0; r < dim - 1; ++r )
                            V[ r ] += face.dirs[ r ][ d ] * ( loc_pts[ op ][ d ] - loc_pts[ pivot ][ d ] );
                    }

                    // store
                    std::array<TF,dim-1> X = solve( M, V );
                    new_inds.push_back( loc_inds[ op ] );
                    new_pts.push_back( X.data() );
                }
            }

            // new_dirs (needed for orientation)
            Pu poly_center = TF( 0 );
            for( TI ind : loc_inds )
                poly_center += ori_pts[ ind ];
            poly_center /= TF( loc_inds.size() );

            Pu face_center = TF( 0 );
            for( TI ind : new_inds )
                face_center += ori_pts[ ind ];
            face_center /= TF( new_inds.size() );

            std::vector<Pu> new_ori_dirs = ori_dirs;
            new_ori_dirs.push_back( face_center - poly_center );

            // register face
            nexts.push_back( { new_pts, new_inds, ori_pts, new_ori_dirs } );
            faces.push_back( std::move( face ) );
        } );
    }

    // sort all the faces
    std::sort( nexts.begin(), nexts.end() );
}

template<int dim,class TF>
ConvexHullIndices<dim,TF>::ConvexHullIndices( const std::vector<Pt> &pts ) : ConvexHullIndices( pts, range<TI>( pts.size() ), pts, {} ) {
}

template<int dim,class TF>
void ConvexHullIndices<dim,TF>::write_to_stream( std::ostream &os, std::string sp ) const {
    os << sp << "--";
    for( const Next &next : nexts )
        next.write_to_stream( os << "\n", sp + "  " );
}

template<int dim,class TF> template<class Pu>
TF ConvexHullIndices<dim,TF>::measure( const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs, TF div ) const {
    TF res = 0;
    Pu C = center( ori_pts );
    for( const Next &next : nexts ) {
        std::vector<Pu> new_dirs = ori_dirs;
        new_dirs.push_back( next.center( ori_pts ) - C );
        res += next.measure( ori_pts, new_dirs, div * ( dim - 1 ) );
    }
    return res;
}

template<int dim,class TF> template<class Pu>
Pu ConvexHullIndices<dim,TF>::center( const std::vector<Pu> &ori_pts ) const {
    Pu res( 0 );
    for( const Next &next : nexts )
        res += next.center( ori_pts );
    return res / TF( nexts.size() + ( nexts.size() == 0 ) );
}

template<int dim,class TF>
bool ConvexHullIndices<dim,TF>::operator<( const ConvexHullIndices &that ) const {
    return nexts < that.nexts;
}

template<int dim,class TF>
bool ConvexHullIndices<dim,TF>::is_a_permutation_of( const std::vector<Pt> &that_pts, const ConvexHullIndices &that, TI *perm ) const {
    // true if a permutation of indices
    if ( nexts.size() != that.nexts.size() || sorted_sizes() != that.sorted_sizes() )
        return false;

    // summary of direct links
    std::vector<std::set<TI>> this_links;
    std::vector<std::set<TI>> that_links;
    this->get_links_rec( this_links );
    that. get_links_rec( that_links );

    // list of possibilities for all the nodes (start with all the possible nodes)
    std::set<TI> base_set;
    for( TI i = 0; i < this_links.size(); ++i )
        base_set.insert( i );
    std::vector<std::set<TI>> possibilities( this_links.size(), base_set ); // nodes of this => nodes of that

    // test all "possible" permutations from this set of possibilities
    return test_permutations( that_pts, that, perm, this_links, that_links, possibilities );
}

template<int dim,class TF>
void ConvexHullIndices<dim,TF>::get_links_rec( std::vector<std::set<TI>> &links ) const {
    for( const Next &next : nexts )
        next.get_links_rec( links );
}

template<int dim,class TF>
typename ConvexHullIndices<dim,TF>::Sizes ConvexHullIndices<dim,TF>::sorted_sizes() const {
    Sizes res;
    for( const Next &next : nexts )
        res.push_back( next.sorted_sizes() );
    std::sort( res.begin(), res.end() );
    return res;
}

template<int dim, class TF>
bool ConvexHullIndices<dim,TF>::test_permutations( const std::vector<Pt> &that_pts, const ConvexHullIndices &that, TI *perm_this_to_that, const std::vector<std::set<TI>> &this_links, const std::vector<std::set<TI>> &that_links, const std::vector<std::set<TI>> &possibilities ) const {
    // test if no remaining possibility, or if fully determined
    TI best_undetermined_node = 0;
    TI best_score = this_links.size() + 1;
    for( TI num_node = 0; num_node < this_links.size(); ++num_node ) {
        if ( possibilities[ num_node ].empty() )
            return false;
        if ( possibilities[ num_node ].size() > 1 && best_score > possibilities[ num_node ].size() ) {
            best_score = possibilities[ num_node ].size();
            best_undetermined_node = num_node;
        }
    }
    if ( best_score > this_links.size() ) {
        // here we have a permutation that get the same edge/node graph => check it
        for( TI num_node_this = 0; num_node_this < this_links.size(); ++num_node_this )
            perm_this_to_that[ num_node_this ] = *possibilities[ num_node_this ].begin();

        ConvexHullIndices this_mod = *this;
        this_mod.replace_inds( perm_this_to_that );
        if ( this_mod.measure( that_pts ) < 0 )
            return false;
        return ! ( this_mod < that || that < this_mod );
    }

    // set value of best_undetermined_node among all the possibilities
    for( TI prop : possibilities[ best_undetermined_node ] ) {
        // update possibilities (best_undetermined_node in this => prop in that)
        std::vector<std::set<TI>> new_possibilities = possibilities;
        for( TI n = 0; n < this_links.size(); ++n ) {
            if ( n == best_undetermined_node ) {
                new_possibilities[ n ] = { prop };
            } else {
                new_possibilities[ n ].erase( prop );
            }
        }

        // si on fixe un noeud, on bloque les noeuds en lien direct
        for( TI linked_this_node : this_links[ best_undetermined_node ] ) {
            std::set<TI> a = std::move( new_possibilities[ linked_this_node ] );
            const std::set<TI> &b = that_links[ prop ];
            std::set_intersection(
                a.begin(), a.end(),
                b.begin(), b.end(),
                std::inserter(
                    new_possibilities[ linked_this_node ],
                    new_possibilities[ linked_this_node ].begin()
                )
            );
        }

        if ( test_permutations( that_pts, that, perm_this_to_that, this_links, that_links, new_possibilities ) )
            return true;
    }

    // not found with this set of possibilities
    return false;
}

template<class TF> template<class Pu>
void ConvexHullIndices<1,TF>::display_vtk( VtkOutput &/*vo*/, const std::vector<Pu> &/*pts*/ ) const {
}

template<int dim,class TF> template<class Pu>
void ConvexHullIndices<dim,TF>::display_vtk( VtkOutput &vo, const std::vector<Pu> &pts ) const {
    if ( dim == 2 ) {
        std::vector<VtkOutput::Pt> npts;
        for( TI p : ordered_pt_seq( N<dim>() ) ) {
            VtkOutput::Pt n( 0 );
            for( TI d = 0; d < Pu::dim; ++d )
                n[ d ] = conv( pts[ p ][ d ], S<VtkOutput::TF>() );
            npts.push_back( n );
        }
        return vo.add_polygon( npts );
    }

    for( const Next &next : nexts )
        next.display_vtk( vo, pts );
}


template<int dim,class TF>
std::vector<typename ConvexHullIndices<dim,TF>::TI> ConvexHullIndices<dim,TF>::ordered_pt_seq( N<2> ) const {
    if ( nexts.empty() )
        return {};
    std::vector<TI> res{ nexts[ 0 ].inds[ 0 ], nexts[ 0 ].inds[ 1 ] };
    while ( true ) {
        for( const Next &next : nexts ) {
            if ( next.inds[ 0 ] != res.back() )
                continue;
            if ( next.inds[ 1 ] == res[ 0 ] )
                return res;
            res.push_back( next.inds[ 1 ] );
        }
    }
}

template<class TF>
void ConvexHullIndices<1,TF>::replace_inds( const TI *repl ) {
    for( TI i = 0; i < inds.size(); ++i )
        inds[ i ] = repl[ inds[ i ] ];
}

template<int dim,class TF>
void ConvexHullIndices<dim,TF>::replace_inds( const TI *repl ) {
    for( Next &next : nexts )
        next.replace_inds( repl );
    std::sort( nexts.begin(), nexts.end() );
}

template<class TF> template<class Pu>
void ConvexHullIndices<1,TF>::on_normal( const std::vector<Pu> &pts, const std::function<void(Pu,Pu)> &f, std::vector<Pu> dirs ) const {
    dirs.push_back( pts[ inds[ 1 ] ] - pts[ inds[ 0 ] ] );
    f( pts[ inds[ 0 ] ], cross_prod( dirs.data() ) );
}

template<int dim,class TF> template<class Pu>
void ConvexHullIndices<dim,TF>::on_normal( const std::vector<Pu> &pts, const std::function<void(Pu,Pu)> &f, std::vector<Pu> dirs ) const {
    dirs.push_back( nexts[ 0 ].center( pts ) - center( pts ) );
    nexts[ 0 ].on_normal( pts, f, dirs );
}

#pragma once

#include "../../Support/ThreadPool.h"
#include "../../Support/BinStream.h"
#include "../../Support/Mpi.h"
#include <Eigen/Cholesky>
#include "SpZGrid.h"
#include <queue>
#include <cmath>
#include <set>

#ifdef WANT_STAT
#include "../../Support/Stat.h"
#endif // WANT_STAT

namespace sdot {

template<class Pc>
SpZGrid<Pc>::SpZGrid( TI max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
    depth_initial_send = 5;
}

template<class Pc>
void SpZGrid<Pc>::update( const Pt *positions, const TF *weights, TI nb_diracs, bool positions_have_changed, bool weights_have_changed, bool clip_at_sqrt_weight ) {
    if ( positions_have_changed || weights_have_changed ) {
        dirac_indices.resize( nb_diracs );
        for( std::size_t i = 0; i < nb_diracs; ++i )
            dirac_indices[ i ] = i;

        boxes.clear();
        boxes.emplace_back();
        root = &boxes.back();
        update_box( positions, weights, root, 0, nb_diracs, 0 );

        //
        initial_send( positions, weights );
    }
}

template<class Pc>
void SpZGrid<Pc>::update_box( const Pt *positions, const TF *weights, Box *box, TI beg_indices, TI end_indices, TI depth ) {
    using TMat = Eigen::Matrix<TF,nb_coeffs_w_approx,nb_coeffs_w_approx>;
    using TVec = Eigen::Matrix<TF,nb_coeffs_w_approx,1>;
    using std::swap;
    using std::pow;
    using std::min;
    using std::max;

    box->beg_indices = beg_indices;
    box->end_indices = end_indices;
    box->depth       = depth;
    box->rank        = mpi->rank();

    TMat M;
    TVec V;
    if ( degree_w_approx > 0 ) {
        for( TI r = 0; r < nb_coeffs_w_approx; ++r ) {
            for( TI c = 0; c < nb_coeffs_w_approx; ++c )
                M.coeffRef( r, c ) = 0;
            V[ r ] = 0;
        }
    }

    // update limits + matrix coeffs
    Pt min_pt( + std::numeric_limits<TF>::max() );
    Pt max_pt( - std::numeric_limits<TF>::max() );
    for( std::size_t num_ind = beg_indices; num_ind < end_indices; ++num_ind ) {
        TI i = dirac_indices[ num_ind ];
        Pt p = positions[ i ];
        min_pt = min( min_pt, p );
        max_pt = max( max_pt, p );

        if ( degree_w_approx > 0 ) {
            std::array<TF,nb_coeffs_w_approx> coeffs;
            coeffs[ 0 ] = 1;
            if ( degree_w_approx >= 1 )
                for( std::size_t d = 0; d < dim; ++d )
                    coeffs[ 1 + d ] = p[ d ];
            if ( degree_w_approx >= 2 )
                for( std::size_t d = 0; d < dim; ++d )
                    for( std::size_t e = 0; e <= d; ++e )
                        coeffs[ 1 + dim + d * ( d + 1 ) / 2 + e ] = p[ d ] * p[ e ];

            TF val = weights[ i ];
            for( TI r = 0; r < nb_coeffs_w_approx; ++r ) {
                for( TI c = 0; c <= r; ++c )
                    M.coeffRef( r, c ) += coeffs[ r ] * coeffs[ c ];
                V[ r ] += coeffs[ r ] * val;
            }
        }
    }
    box->min_pt = min_pt;
    box->max_pt = max_pt;

    Pt lim_pt = TF( 0.5 ) * ( min_pt + max_pt );
    auto num_ch = [&]( Pt p ) {
        TI bi = 0;
        for( std::size_t d = 0; d < dim; ++d )
            bi += ( 1 << d ) * ( p[ d ] >= lim_pt[ d ] );
        return bi;
    };

    // compute coeffs
    if ( degree_w_approx > 0 ) {
        for( TI c = 0; c < nb_coeffs_w_approx; ++c )
            for( TI r = 0; r < c; ++r )
                M.coeffRef( r, c ) = M.coeffRef( c, r );
        TF ad = 1e-10 * M.diagonal().maxCoeff();
        for( TI c = 0; c < nb_coeffs_w_approx; ++c )
            M.coeffRef( c, c ) += ad;


        Eigen::LLT<TMat> llt;
        llt.compute( M );
        V = llt.solve( V );
    }

    // update height + nb points in each sub box
    constexpr TI nb_ch = ( 1 << dim );
    std::array<TI,nb_ch> sb_end;
    for( TI &v : sb_end )
        v = 0;
    V[ 0 ] = - std::numeric_limits<TF>::max();
    for( std::size_t num_ind = beg_indices; num_ind < end_indices; ++num_ind ) {
        TI i = dirac_indices[ num_ind ];
        Pt p = positions[ i ];

        TF val = weights[ i ];
        if ( degree_w_approx >= 1 )
            for( std::size_t d = 0; d < dim; ++d )
                val -= V[ 1 + d ] * p[ d ];
        if ( degree_w_approx >= 2 )
            for( std::size_t d = 0; d < dim; ++d )
                for( std::size_t e = 0; e <= d; ++e )
                    val -= V[ 1 + dim + d * ( d + 1 ) / 2 + e ] * p[ d ] * p[ e ];
        V[ 0 ] = max( V[ 0 ], val );

        ++sb_end[ num_ch( p ) ];
    }

    // save coeffs
    for( TI r = 0; r < nb_coeffs_w_approx; ++r )
        box->coeffs_w_approx[ r ] = V[ r ];

    // inplace sorting of dirac indices (first box, second one, ...)
    std::array<TI,nb_ch> sb_beg;
    for( TI n = 0, a = beg_indices; n < nb_ch; ++n ) {
        sb_beg[ n ] = a;
        sb_end[ n ] = ( a += sb_end[ n ] );
    }
    for( TI ni = 0; ni < nb_ch; ++ni ) {
        if ( sb_beg[ ni ] == sb_end[ ni ] )
            continue;
        while ( true ) {
            TI &i = dirac_indices[ sb_beg[ ni ] ];
            TI bi = num_ch( positions[ i ] );
            if ( bi != ni )
                swap( i, dirac_indices[ sb_beg[ bi ]++ ] );
            else if ( ++sb_beg[ ni ] == sb_end[ ni ] )
                break;
        }
    }

    // recursion
    box->last_child = nullptr;
    if ( end_indices - beg_indices > max_diracs_per_cell ) {
        for( TI n = 0; n < nb_ch; ++n ) {
            TI beg = n ? sb_beg[ n - 1 ] : beg_indices;
            TI end = sb_beg[ n ];
            if ( beg == end )
                continue;

            boxes.emplace_back();
            Box *ch = &boxes.back();

            ch->sibling = box->last_child;
            box->last_child = ch;

            update_box( positions, weights, ch, beg, end, depth + 1 );
        }
    }
    #ifdef WANT_STAT
    else {
        stat.add( "depth", depth );
    }
    #endif // WANT_STAT
}

template<class Pc>
typename SpZGrid<Pc>::TI SpZGrid<Pc>::nb_diracs( Box *box ) {
    if ( ! box )
        return 0;
    if ( box->last_child )
        return nb_diracs( box->last_child ) +
               nb_diracs( box->sibling );
    return box->end_indices - box->end_indices + box->ext_pwi.size() +
           nb_diracs( box->sibling );
}

template<class Pc>
std::string SpZGrid<Pc>::ext_info() const {
    std::ostringstream ss;
    for( const Neighbor &ng : neighbors )
        ss << "\n  r=" << ng.mpi_rank << " c=" << nb_diracs( ng.root );
    return ss.str();
}

template<class Pc>
std::vector<char> SpZGrid<Pc>::serialize_rec( const Pt *positions, const TF *weights, std::vector<Box *> front, TI max_depth, N<0> ) {
    TODO;
    return {};
}

template<class Pc>
std::vector<char> SpZGrid<Pc>::serialize_rec( const Pt *positions, const TF *weights, std::vector<Box *> front, TI max_depth, N<1> ) {
    Hpipe::CbQueue cq;
    std::size_t num_in_front = 0;
    Hpipe::BinStream<Hpipe::CbQueue> bq( &cq );
    while ( num_in_front < front.size() ) {
        Box *box = front[ num_in_front++ ];

        bq << box->coeffs_w_approx;
        bq << box->min_pt;
        bq << box->max_pt;
        bq.write_unsigned( box->depth );

        if ( box->depth < max_depth && box->last_child ) {
            bq.write_unsigned( front.size() );
            front.push_back( box->last_child );
        } else
            bq.write_unsigned( 0u );

        if ( box->sibling ) {
            bq.write_unsigned( front.size() );
            front.push_back( box->sibling );
        } else
            bq.write_unsigned( 0u );

        // if leaf, send the diracs
        if ( box->last_child == nullptr ) {
            bq.write_unsigned( box->end_indices - box->beg_indices );
            for( TI num_ind = box->beg_indices; num_ind < box->end_indices; ++num_ind ) {
                TI num_dirac = dirac_indices[ num_ind ];
                bq << positions[ num_dirac ];
                bq << weights[ num_dirac ];
                bq.write_unsigned( num_dirac );
            }
        } else
            bq.write_unsigned( 0u );
    }

    std::vector<char> src( cq.size() );
    cq.read_some( src.data(), src.size() );
    return src;
}

template<class Pc>
typename SpZGrid<Pc>::Box* SpZGrid<Pc>::deserialize_rec( const std::vector<char> &dst, int ext_rank, N<0> ) {
    TODO;
    return 0;
}

template<class Pc>
typename SpZGrid<Pc>::Box* SpZGrid<Pc>::deserialize_rec( const std::vector<char> &dst, int ext_rank, N<1> ) {
    Hpipe::CmString cm( dst.data(), dst.size() );
    Hpipe::BinStream<Hpipe::CmString> bq( &cm );

    std::vector<Box *> new_boxes;
    while ( ! bq.empty() ) {
        boxes.emplace_back();
        Box *box = &boxes.back();
        new_boxes.push_back( box );

        box->beg_indices = 0;
        box->end_indices = 0;
        box->last_child  = nullptr;
        box->sibling     = nullptr;
        box->rank        = ext_rank;

        bq >> box->coeffs_w_approx;
        bq >> box->min_pt;
        bq >> box->max_pt;
        bq >> box->depth;

        box->last_child_index = bq.read_unsigned();
        box->sibling_index = bq.read_unsigned();

        std::size_t nb_ext_diracs = bq.read_unsigned();
        box->ext_pwi.resize( nb_ext_diracs );
        for( TI num_ext_dirac = 0; num_ext_dirac < nb_ext_diracs; ++num_ext_dirac ) {
            PWI &pwi = box->ext_pwi[ num_ext_dirac ];
            bq >> pwi.position;
            bq >> pwi.weight;
            bq >> pwi.num_dirac;
        }
    }

    for( Box *box : new_boxes ) {
        if ( box->last_child_index )
            box->last_child = new_boxes[ box->last_child_index ];
        if ( box->sibling_index )
            box->sibling = new_boxes[ box->sibling_index ];
    }

    return new_boxes.size() ? new_boxes[ 0 ] : nullptr;
}

template<class Pc>
void SpZGrid<Pc>::initial_send( const Pt *positions, const TF *weights ) {
    neighbors.clear();

    // send a serialized shallow repr of the grid
    if ( mpi->size() > 1 ) {
        std::vector<std::vector<char>> dst;
        std::vector<char> src = serialize_rec( positions, weights, { root }, depth_initial_send, TFIsStd() );
        mpi->all_gather( dst, src.data(), src.size() );

        // deserialize
        neighbors.reserve( mpi->size() );
        for( int i = 0; i < (int)dst.size(); ++i )
            if ( i != mpi->rank() )
                if ( Box *ext_root = deserialize_rec( dst[ i ], i, TFIsStd() ) )
                    neighbors.push_back( { i, ext_root } );
    }
}

template<class Pc> template<class V>
void SpZGrid<Pc>::display( V &vtk_output, TF z ) const {
    std::deque<Box*> front;
    front.push_back( root );
    while ( front.size() ) {
        Box* box = front.back();
        front.pop_back();

        switch ( dim ) {
        case 2:
            vtk_output.add_lines( {
                Point3<TF>{ box->min_pt[ 0 ], box->min_pt[ 1 ], z * box->depth },
                Point3<TF>{ box->max_pt[ 0 ], box->min_pt[ 1 ], z * box->depth },
                Point3<TF>{ box->max_pt[ 0 ], box->max_pt[ 1 ], z * box->depth },
                Point3<TF>{ box->min_pt[ 0 ], box->max_pt[ 1 ], z * box->depth },
                Point3<TF>{ box->min_pt[ 0 ], box->min_pt[ 1 ], z * box->depth },
            }, { TF( box->depth ) } );
            break;
        case 3:
            TODO;
            break;
        default:
            TODO;
        }

        for( Box* ch = box->last_child; ch; ch = ch->sibling )
            front.push_back( ch );
    }
}

template<class Pc>
int SpZGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, TI num )> &cb, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc ) {
    return for_each_laguerre_cell( [&]( CP &cp, TI num, int ) {
        cb( cp, num );
    }, starting_lc, positions, weights, nb_diracs, stop_if_void_lc );
}

template<class Pc>
int SpZGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, TI num, int num_thread )> &cb, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc, bool ball_cut ) {
    return ball_cut ?
        for_each_laguerre_cell( cb, starting_lc, positions, weights, nb_diracs, stop_if_void_lc, N<1>() ) :
        for_each_laguerre_cell( cb, starting_lc, positions, weights, nb_diracs, stop_if_void_lc, N<0>() ) ;
}


template<class Pc>
float SpZGrid<Pc>::Box::dist_2( Pt p, TF w ) const {
    return float( norm_2_p2( p - TF( 0.5 ) * ( min_pt + max_pt ) ) );
}

template<class Pc> template<class TA>
typename SpZGrid<Pc>::TF SpZGrid<Pc>::w_approx( const TA &c, Pt x ) const {
    TF res = c[ 0 ];
    if ( degree_w_approx >= 1 )
        for( std::size_t i = 0; i < dim; ++i )
            res += c[ 1 + i ] * x[ i ];
    if ( degree_w_approx >= 2 )
        for( std::size_t i = 0, cpt = 0; i < dim; ++i )
            for( std::size_t j = 0; j <= i; ++j, ++cpt )
                res += c[ 1 + dim + cpt ] * x[ i ] * x[ j ];
    return res;
};

template<class Pc> template<class Node>
bool SpZGrid<Pc>::can_be_evicted( CP &lc, Pt c0, TF w0, Box *box, int num_sym, std::vector<Node *> &front ) {
    using std::pow;
    using std::min;
    using std::max;

    Pt min_pt = sym( box->min_pt, num_sym );
    Pt max_pt = sym( box->max_pt, num_sym );

    if ( CP::keep_min_max_coords ) {
        TF g_min = w0 - box->coeffs_w_approx[ 0 ];
        for( int d = 0; d < dim; ++d ) {
            TF d_min = std::numeric_limits<TF>::max();

            //
            TF mim = max( min_pt[ d ], lc.min_coord[ d ] );
            TF mam = min( max_pt[ d ], lc.max_coord[ d ] );
            if ( mam >= mim ) {
                TF pmi = pow( c0[ d ] - mim, 2 );
                TF pma = pow( c0[ d ] - mam, 2 );
                d_min = - max( pmi, pma );
            }

            //
            TF mil = lc.min_coord[ d ];
            TF mal = min( min_pt[ d ], lc.max_coord[ d ] );
            if ( mal >= mil ) {
                TF di = c0[ d ] - min_pt[ d ];
                d_min = min( d_min, pow( min_pt[ d ], 2 ) + 2 * di * ( di > 0 ? mil : mal ) - pow( c0[ d ], 2 ) );
            }

            //
            TF mir = max( max_pt[ d ], lc.min_coord[ d ] );
            TF mar = lc.max_coord[ d ];
            if ( mar >= mir ) {
                TF di = c0[ d ] - max_pt[ d ];
                d_min = min( d_min, pow( max_pt[ d ], 2 ) + 2 * di * ( di > 0 ? mir : mar ) - pow( c0[ d ], 2 ) );
            }

            g_min += d_min;
        }

        return g_min > 0;
    }

    //    if ( dim == 3 && c0[ 0 ] >= min_pt[ 0 ] && c0[ 1 ] >= min_pt[ 1 ] && c0[ 2 ] >= min_pt[ 2 ] && c0[ 0 ] <= max_pt[ 0 ] && c0[ 1 ] <= max_pt[ 1 ] && c0[ 2 ] <= max_pt[ 2 ] )
    //        return false;

    //
    //    if ( lc.nodes.empty() )
    //        return true;

    //    Node *node = &*lc.nodes.begin();
    //    node->op_count = ++lc.op_count;

    //    front.clear();
    //    front.push_back( node );

    //    while ( ! front.empty() ) {
    //        Node *node = front.back();
    //        front.pop_back();

    //        // the current node may be invalid ?
    //        Pt c1;
    //        for( int d = 0; d < dim; ++d )
    //            c1[ d ] = min( max_pt[ d ], max( min_pt[ d ], node->pos[ d ] ) );
    //        if ( norm_2_p2( c0 - node->pos ) - w0 > norm_2_p2( c1 - node->pos ) - w_approx( box->coeffs_w_approx, inv_sym( c1, num_sym ) ) )
    //            return false;

    //        // we take a new node if it is able to improve the criterion for some x1 in box
    //        for( const auto &edge : node->edges ) {
    //            if ( edge.n1->op_count == lc.op_count )
    //                continue;
    //            edge.n1->op_count = lc.op_count;

    //            TF dv = 0;
    //            for( int d = 0; d < dim; ++d ) {
    //                TF pi = node->pos[ d ] - edge.n1->pos[ d ];
    //                dv += ( ( pi > 0 ? box->min_pt[ d ] : box->max_pt[ d ] ) - c0[ d ] ) * pi;
    //            }
    //            if ( dv < 0 )
    //                front.push_back( edge.n1 );
    //        }
    //    }

    return lc.all_pos( [&]( Pt pos ) {
        Pt c1;
        for( int d = 0; d < dim; ++d )
            c1[ d ] = min( max_pt[ d ], max( min_pt[ d ], pos[ d ] ) );
        return norm_2_p2( c0 - pos ) - w0 <= norm_2_p2( c1 - pos ) - w_approx( box->coeffs_w_approx, inv_sym( c1, num_sym ) );
    } );

    //    Pt dir = 0.5 * ( min_pt + max_pt ) - c0;
    //    Node *node = lc.find_node_maximizing( [&]( TF &value, Pt pos ) {
    //        value = dot( pos, dir );
    //        return false;
    //    }, false );
    //    Pt c1;
    //    for( int d = 0; d < dim; ++d )
    //        c1[ d ] = min( max_pt[ d ], max( min_pt[ d ], node->pos[ d ] ) );
    //    return norm_2_p2( c0 - node->pos ) - w0 <= norm_2_p2( c1 - node->pos ) - w_approx( box->coeffs_w_approx, inv_sym( c1, num_sym ) );
}

template<class Pc> template<int ball_cut>
int SpZGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, TI num, int num_thread )> &cb, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc, N<ball_cut> ) {
    using std::sqrt;
    using std::pow;
    using std::min;

    struct BoxDistAndNumSym {
        bool  operator<( const BoxDistAndNumSym &that ) const { return dist > that.dist; }
        Box*  box;
        float dist;
        int   num_sym; ///< num of translation/symetry or -1 if untouched
    };

    auto plane_cut = [&]( CP &lc, Pt c0, TF w0, Pt c1, TF w1, TI i1 ) {
        Pt V = c1 - c0;
        TF n = norm_2_p2( V );
        TF x = TF( 0.5 ) + TF( 0.5 ) * ( w0 - w1 ) / n;

        #ifdef PD_WANT_STAT
        stat.add_for_dist( "nb nodes before cut", lc.nb_points() );

        static std::ofstream fout( "cuts.txt" );
        static std::mutex mut;
        mut.lock();
        fout << stat.num_phase << " " << c0 << " " << c0 + x * V << " " << V << "\n";
        mut.unlock();
        #endif

        lc.plane_cut( c0 + x * V, V, i1, N<0>() );
    };

    auto make_lc_and_call_cb = [&]( std::set<std::size_t> &g_missing_ranks, std::vector<TI> &g_interrupted_num_diracs, int &err, std::size_t nb_jobs, int nb_threads, const TI *dirac_indices, TI nb_dirac_indices, int phase ) {
        std::vector<std::vector<TI>> interrupted_num_diracs( nb_threads );
        std::vector<std::set<std::size_t>> missing_ranks( nb_threads );
        std::vector<CP> starting_lcs( nb_threads );
        for( CP &cp : starting_lcs )
            cp = starting_lc;

        thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
            std::priority_queue<BoxDistAndNumSym> front;
            using Node = typename CP::Node;
            std::vector<Node *> front_node;
            CP lc;

            TI beg_num_in_ind = ( num_job + 0 ) * nb_dirac_indices / nb_jobs;
            TI end_num_in_ind = ( num_job + 1 ) * nb_dirac_indices / nb_jobs;
            for( TI num_ind = beg_num_in_ind; num_ind < end_num_in_ind; ++num_ind ) {
                TI num_dirac_0 = dirac_indices[ num_ind ];
                const Pt c0 = positions[ num_dirac_0 ];
                const TF w0 = weights[ num_dirac_0 ];
                if ( err )
                    break;

                // start of lc: cut with nodes in the same cell
                lc = starting_lcs[ num_thread ];

                // init of the front
                if ( Box *ch = root->last_child ) {
                    do {
                        front.push( { ch, ch->dist_2( c0, w0 ), -1 } );
                    } while (( ch = ch->sibling ));
                } else {
                    front.push( { root, 0.0f, -1 } );
                }

                if ( allow_translations ) {
                    for( std::size_t num_sym = 0; num_sym < translations.size(); ++num_sym )
                        front.push( { root, root->dist_2( inv_sym( c0, num_sym ), w0 ), int( num_sym ) } );
                }

                if ( allow_mpi ) {
                    for( const Neighbor &ng : neighbors ) {
                        front.push( { ng.root, ng.root->dist_2( c0, w0 ), -1 } );

                        for( std::size_t num_sym = 0; num_sym < translations.size(); ++num_sym )
                            front.push( { ng.root, ng.root->dist_2( inv_sym( c0, num_sym ), TF( 0 ) ), int( num_sym ) } );
                    }
                }

                // recursive traversal
                bool interrupted = false;
                #ifdef WANT_STAT
                double nb_seen_boxes = 0;
                bool seen_containing_box = false;
                bool used_another_box = false;
                #endif // WANT_STAT
                while ( ! front.empty() ) {
                    int num_sym = front.top().num_sym;
                    Box *box = front.top().box;
                    front.pop();

                    #ifdef WANT_STAT
                    ++nb_seen_boxes;
                    // stat.add( "eviction", can_be_evicted( lc, c0, w0, box, num_sym ) );
                    //                    if ( can_be_evicted( lc, c0, w0, box, num_sym ) )
                    //                        stat.add( "nb_nodes_during_pc", lc.nodes.size() );
                    #endif // WANT_STAT

                    if ( can_be_evicted( lc, c0, w0, box, num_sym, front_node ) )
                        continue;

                    #ifdef WANT_STAT
                    if ( seen_containing_box )
                        used_another_box = true;
                    #endif // WANT_STAT

                    if ( Box* ch = box->last_child ) {
                        do {
                            front.push( { ch, float( ch->dist_2( inv_sym( c0, num_sym ), w0 ) ), num_sym } );
                        } while (( ch = ch->sibling ));
                    } else if ( box->beg_indices < box->end_indices ) {
                        #ifdef WANT_STAT
                        seen_containing_box = true;
                        #endif // WANT_STAT
                        for( TI num_in_ind_1 = box->beg_indices; num_in_ind_1 < box->end_indices; ++num_in_ind_1 ) {
                            TI num_dirac_1 = this->dirac_indices[ num_in_ind_1 ];
                            if ( num_dirac_0 != num_dirac_1 || num_sym >= 0 )
                                plane_cut( lc, c0, w0, sym( positions[ num_dirac_1 ], num_sym ), weights[ num_dirac_1 ], num_dirac_1 + ( num_sym + 1 ) * nb_diracs );
                        }
                    } else if ( allow_mpi && box->ext_pwi.size() ) {
                        for( TI num_in_ext = 0; num_in_ext < box->ext_pwi.size(); ++num_in_ext )
                            plane_cut( lc, c0, w0, sym( box->ext_pwi[ num_in_ext ].position, num_sym ), box->ext_pwi[ num_in_ext ].weight, box->ext_pwi[ num_in_ext ].num_dirac );
                    } else if ( allow_mpi ) {
                        interrupted_num_diracs[ num_thread ].push_back( num_dirac_0 );
                        missing_ranks[ num_thread ].insert( box->rank );
                        interrupted = true;
                        while ( ! front.empty() )
                            front.pop();
                        break;
                    }
                }
                #ifdef WANT_STAT
                stat.add( "nb_seen_boxes", nb_seen_boxes );
                stat.add( "used_another_box", used_another_box );
                #endif // WANT_STAT

                if ( allow_mpi && interrupted )
                    continue;

                //
                if ( ball_cut )
                    lc.ball_cut( positions[ num_dirac_0 ], sqrt( weights[ num_dirac_0 ] ), num_dirac_0 );
                else
                    lc.sphere_center = c0;

                if ( stop_if_void_lc && lc.empty() ) {
                    err = 1;
                    return;
                }

                cb( lc, num_dirac_0, num_thread );
            }
        } );

        for( int i = 0; i < nb_threads; ++i ) {
            g_interrupted_num_diracs.insert( g_interrupted_num_diracs.end(), interrupted_num_diracs[ i ].begin(), interrupted_num_diracs[ i ].end() );
            g_missing_ranks.insert( missing_ranks[ i ].begin(), missing_ranks[ i ].end() );
        }
    };

    // try with information currently available
    int err = 0;
    std::set<std::size_t> missing_ranks;
    std::vector<TI> interrupted_num_diracs;
    int nb_threads = thread_pool.nb_threads(), nb_jobs = 4 * nb_threads;
    make_lc_and_call_cb( missing_ranks, interrupted_num_diracs, err, nb_jobs, nb_threads, dirac_indices.data(), dirac_indices.size(), 0 );
    if ( err )
        return err;

    //
    if ( allow_mpi && mpi->size() > 1 ) {
        for( int phase = 1; ; ++phase ) {
            PMPI( interrupted_num_diracs.size() );

            // get the needs for each machine
            std::vector<std::vector<int>> needs;
            std::vector<int> v_missing_ranks( missing_ranks.begin(), missing_ranks.end() );
            mpi->all_gather( needs, v_missing_ranks.data(), v_missing_ranks.size() );

            // break if no need for more information ?
            bool n = false;
            for( auto &nv : needs ) {
                if ( ! nv.empty() ) {
                    n = true;
                    break;
                }
            }
            if ( n == false )
                break;

            // serialize the tree for each machine that has to send something
            std::vector<char> serialized;
            for( const std::vector<int> &pn : needs ) {
                if ( std::find( pn.begin(), pn.end(), mpi->rank() ) != pn.end() ) {
                    serialized = serialize_rec( positions, weights, { root }, 100000, TFIsStd() );
                    break;
                }
            }

            // selective send
            std::vector<std::vector<char>> ext;
            mpi->selective_send_and_recv( ext, needs, serialized );

            // update the neighbor info
            for( std::size_t num_in_v_missing_rank = 0; num_in_v_missing_rank < v_missing_ranks.size(); ++num_in_v_missing_rank ) {
                int missing_rank = v_missing_ranks[ num_in_v_missing_rank ];
                if ( Box *box = deserialize_rec( ext[ num_in_v_missing_rank ], missing_rank, TFIsStd() ) ) {
                    for( Neighbor &ng : neighbors ) {
                        if ( ng.mpi_rank == missing_rank ) {
                            ng.root = box;
                            break;
                        }
                    }
                }
            }

            // try to make the missing cells
            missing_ranks.clear();
            std::vector<TI> old_interrupted_num_diracs = std::move( interrupted_num_diracs );
            make_lc_and_call_cb( missing_ranks, interrupted_num_diracs, err, nb_jobs, nb_threads, old_interrupted_num_diracs.data(), old_interrupted_num_diracs.size(), phase );
            if ( err )
                return err;
        }
    }

    return err;
}


} // namespace sdot


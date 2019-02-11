#include "../system/ThreadPool.h"
#include "../system/BinStream.h"
#include "../system/Mpi.h"
#include <eigen3/Eigen/Cholesky>
#include "SpZGrid.h"
#include <queue>
#include <cmath>
#include <set>

namespace sdot {

template<class Pc>
SpZGrid<Pc>::SpZGrid( TI max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
    depth_initial_send = 5;
}

template<class Pc>
void SpZGrid<Pc>::update( const Pt *positions, const TF *weights, TI nb_diracs, bool positions_have_changed, bool weights_have_changed ) {
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
    for( TI r = 0; r < nb_coeffs_w_approx; ++r ) {
        for( TI c = 0; c < nb_coeffs_w_approx; ++c )
            M.coeffRef( r, c ) = 0;
        V[ r ] = 0;
    }

    // update limits + matrix coeffs
    Pt min_pt( + std::numeric_limits<TF>::max() );
    Pt max_pt( - std::numeric_limits<TF>::max() );
    for( std::size_t num_ind = beg_indices; num_ind < end_indices; ++num_ind ) {
        TI i = dirac_indices[ num_ind ];
        Pt p = positions[ i ];
        min_pt = min( min_pt, p );
        max_pt = max( max_pt, p );

        std::array<TF,nb_coeffs_w_approx> coeffs;
        coeffs[ 0 ] = 1;
        if ( degree_w_approx >= 1 )
            for( std::size_t d = 0; d < dim; ++d )
                coeffs[ 1 + d ] = p[ d ];
        if ( degree_w_approx >= 2 )
            for( std::size_t d = 0; d < dim; ++d )
                for( std::size_t e = 0; e <= d; ++e )
                    coeffs[ 1 + dim + d * ( d + 1 ) / 2 + e ] = p[ d ] * p[ e ];

        TF val = norm_2_p2( p ) - weights[ i ];
        for( TI r = 0; r < nb_coeffs_w_approx; ++r ) {
            for( TI c = 0; c <= r; ++c )
                M.coeffRef( r, c ) += coeffs[ r ] * coeffs[ c ];
            V[ r ] += coeffs[ r ] * val;
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
    for( TI c = 0; c < nb_coeffs_w_approx; ++c )
        for( TI r = 0; r < c; ++r )
            M.coeffRef( r, c ) = M.coeffRef( c, r );
    TF ad = 1e-10 * M.diagonal().maxCoeff();
    for( TI c = 0; c < nb_coeffs_w_approx; ++c )
        M.coeffRef( c, c ) += ad;

    Eigen::LLT<TMat> llt;
    llt.compute( M );
    V = llt.solve( V );

    // update height + nb points in each sub box
    constexpr TI nb_ch = ( 1 << dim );
    std::array<TI,nb_ch> sb_end;
    for( TI &v : sb_end )
        v = 0;
    V[ 0 ] = + std::numeric_limits<TF>::max();
    for( std::size_t num_ind = beg_indices; num_ind < end_indices; ++num_ind ) {
        TI i = dirac_indices[ num_ind ];
        Pt p = positions[ i ];

        TF val = norm_2_p2( p ) - weights[ i ];
        if ( degree_w_approx >= 1 )
            for( std::size_t d = 0; d < dim; ++d )
                val -= V[ 1 + d ] * p[ d ];
        if ( degree_w_approx >= 2 )
            for( std::size_t d = 0; d < dim; ++d )
                for( std::size_t e = 0; e <= d; ++e )
                    val -= V[ 1 + dim + d * ( d + 1 ) / 2 + e ] * p[ d ] * p[ e ];
        V[ 0 ] = min( V[ 0 ], val );

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
std::vector<char> SpZGrid<Pc>::serialize_rec( const Pt *positions, const TF *weights, std::vector<Box *> front, int max_depth ) {
    Hpipe::CbQueue cq;
    std::size_t num_in_front = 0;
    Hpipe::BinStream<Hpipe::CbQueue> bq( &cq );
    while ( num_in_front < front.size() ) {
        Box *box = front[ num_in_front++ ];

        bq << box->coeffs_w_approx;
        bq << box->min_pt;
        bq << box->max_pt;
        bq << box->depth;

        if ( box->depth < max_depth && box->last_child ) {
            bq << front.size();
            front.push_back( box->last_child );
        } else
            bq << 0u;

        if ( box->sibling ) {
            bq << front.size();
            front.push_back( box->sibling );
        } else
            bq << 0u;

        // if leaf, send the diracs
        if ( box->last_child == nullptr ) {
            bq << box->end_indices - box->beg_indices;
            for( TI num_ind = box->beg_indices; num_ind < box->end_indices; ++num_ind ) {
                TI num_dirac = dirac_indices[ num_ind ];
                bq << positions[ num_dirac ];
                bq << weights[ num_dirac ];
                bq << num_dirac;
            }
        } else
            bq << 0u;
    }

    std::vector<char> src( cq.size() );
    cq.read_some( src.data(), src.size() );
    return src;
}

template<class Pc>
typename SpZGrid<Pc>::Box* SpZGrid<Pc>::deserialize_rec( const std::vector<char> &dst, int ext_rank ) {
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

        reinterpret_cast<std::size_t&>( box->last_child ) = bq.read_unsigned();
        reinterpret_cast<std::size_t&>( box->sibling    ) = bq.read_unsigned();

        std::size_t nb_ext_diracs = bq.read();
        box->ext_pwi.resize( nb_ext_diracs );
        for( TI num_ext_dirac = 0; num_ext_dirac < nb_ext_diracs; ++num_ext_dirac ) {
            PWI &pwi = box->ext_pwi[ num_ext_dirac ];
            bq >> pwi.position;
            bq >> pwi.weight;
            bq >> pwi.num_dirac;
        }
    }

    for( Box *box : new_boxes ) {
        if ( auto l = reinterpret_cast<const std::size_t&>( box->last_child ) )
            box->last_child = new_boxes[ l ];
        if ( auto l = reinterpret_cast<const std::size_t&>( box->sibling    ) )
            box->sibling    = new_boxes[ l ];
    }

    return new_boxes.size() ? new_boxes[ 0 ] : nullptr;
}

template<class Pc>
void SpZGrid<Pc>::initial_send( const Pt *positions, const TF *weights ) {
    // send a serialized shallow repr of the grid
    std::vector<std::vector<char>> dst;
    std::vector<char> src = serialize_rec( positions, weights, { root }, depth_initial_send );
    mpi->all_gather( dst, src.data(), src.size() );

    // deserialize
    neighbors.clear();
    neighbors.reserve( mpi->size() );
    for( std::size_t i = 0; i < dst.size(); ++i )
        if ( i != mpi->rank() )
            if ( Box *ext_root = deserialize_rec( dst[ i ], i ) )
                neighbors.push_back( { i, ext_root } );
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

    auto sym = [&]( Pt pt, int num_sym ) {
        if ( allow_translations && num_sym >= 0 )
            return pt + translations[ num_sym ];
        return pt;
    };

    auto inv_sym = [&]( Pt pt, int num_sym ) {
        if ( allow_translations && num_sym >= 0 )
            return pt - translations[ num_sym ];
        return pt;
    };

    auto plane_cut = [&]( CP &lc, Pt c0, TF w0, Pt c1, TF w1, TI i1 ) {
        Pt V = c1 - c0;
        TF n = norm_2_p2( V );
        TF x = TF( 0.5 ) + TF( 0.5 ) * ( w0 - w1 ) / n;
        lc.plane_cut( c0 + x * V, V, i1, N<0>() );
    };

    auto can_be_evicted = [&]( CP &lc, Pt c0, TF w0, Box *box, int num_sym ) -> bool {
        const auto &c = box->coeffs_w_approx;
        c0 = inv_sym( c0, num_sym );

        auto pol_val_2 = [&]( Pt p, TF x, TF y ) {
            if ( degree_w_approx >= 2 )
                return c[ 0 ] + ( c[ 1 ] - 2 * p.x ) * x + ( c[ 2 ] - 2 * p.y ) * y + c[ 3 ] * pow( x, 2 ) + c[ 4 ] * x * y + c[ 5 ] * pow( y, 2 );
            if ( degree_w_approx >= 1 )
                return c[ 0 ] + ( c[ 1 ] - 2 * p.x ) * x + ( c[ 2 ] - 2 * p.y ) * y;
            return c[ 0 ];
        };

        for( TI num_p = 0; num_p < lc.nb_points; ++num_p ) {
            Pt p = inv_sym( lc.point( num_p ), num_sym );
            TF cm = norm_2_p2( c0 ) - w0 - 2 * dot( c0, p );

            if ( degree_w_approx == 0 && cm > pol_val_2( p, 0, 0 ) )
                return false;

            if ( degree_w_approx >= 1 ) {
                // corners
                if ( cm > pol_val_2( p, box->min_pt.x, box->min_pt.y ) )
                    return false;
                if ( cm > pol_val_2( p, box->max_pt.x, box->min_pt.y ) )
                    return false;
                if ( cm > pol_val_2( p, box->min_pt.x, box->max_pt.y ) )
                    return false;
                if ( cm > pol_val_2( p, box->max_pt.x, box->max_pt.y ) )
                    return false;

                // lines
                TF b_x, b_y;
                if ( c[ 3 ] ) {
                    // y = box->min_pt.y
                    b_x = 0.5 * ( 2 * p.x - c[ 1 ] - c[ 4 ] * box->min_pt.y ) / c[ 3 ];
                    if ( b_x > box->min_pt.x && b_x < box->max_pt.x && cm > pol_val_2( p, b_x, box->min_pt.y ) )
                        return false;

                    // y = box->max_pt.y
                    b_x = 0.5 * ( 2 * p.x - c[ 1 ] - c[ 4 ] * box->max_pt.y ) / c[ 3 ];
                    if ( b_x > box->min_pt.x && b_x < box->max_pt.x && cm > pol_val_2( p, b_x, box->max_pt.y ) )
                        return false;
                }
                if ( c[ 5 ] ) {
                    // x = box->min_pt.x
                    b_y = 0.5 * ( 2 * p.y - c[ 2 ] - c[ 4 ] * box->min_pt.x ) / c[ 5 ];
                    if ( b_y > box->min_pt.y && b_y < box->max_pt.y && cm > pol_val_2( p, box->min_pt.x, b_y ) )
                        return false;

                    // x = box->max_pt.x
                    b_y = 0.5 * ( 2 * p.y - c[ 2 ] - c[ 4 ] * box->max_pt.x ) / c[ 5 ];
                    if ( b_y > box->min_pt.y && b_y < box->max_pt.y && cm > pol_val_2( p, box->max_pt.x, b_y ) )
                        return false;
                }
            }

            // surface
            if ( degree_w_approx >= 2 ) {
                TF det = 4 * c[ 3 ] * c[ 5 ] - pow( c[ 4 ], 2 );
                if ( det ) {
                    TF b_x = ( 2 * ( 2 * p.x - c[ 1 ] ) * c[ 5 ] - ( 2 * p.y - c[ 2 ] ) * c[ 4 ] ) / det;
                    if ( b_x > box->min_pt.x && b_x < box->max_pt.x ) {
                        TF b_y = ( 2 * ( 2 * p.y - c[ 2 ] ) * c[ 3 ] - ( 2 * p.x - c[ 1 ] ) * c[ 4 ] ) / det;
                        if ( b_y > box->min_pt.y && b_y < box->max_pt.y && cm > pol_val_2( p, b_x, b_y ) )
                            return false;
                    }
                }
            }
        }

        return true;
    };

    auto make_lc_and_call_cb = [&]( std::set<std::size_t> &g_missing_ranks, std::vector<TI> &g_interrupted_num_diracs, int &err, std::size_t nb_jobs, int nb_threads, const TI *dirac_indices, TI nb_dirac_indices, int phase ) {
        std::vector<std::vector<TI>> interrupted_num_diracs( nb_threads );
        std::vector<std::set<std::size_t>> missing_ranks( nb_threads );

        thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
            std::priority_queue<BoxDistAndNumSym> front;
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
                lc = starting_lc;

                // init of the front
                if ( Box* ch = root->last_child ) {
                    do {
                        front.push( { ch, ch->dist_2( c0, w0 ), -1 } );
                    } while (( ch = ch->sibling ));
                } else {
                    front.push( { root, 0.0f, -1 } );
                }
                for( std::size_t num_sym = 0; num_sym < translations.size(); ++num_sym )
                    front.push( { root, root->dist_2( inv_sym( c0, num_sym ), w0 ), int( num_sym ) } );

                if ( allow_mpi ) {
                    for( const Neighbor &ng : neighbors ) {
                        front.push( { ng.root, ng.root->dist_2( c0, w0 ), -1 } );

                        for( std::size_t num_sym = 0; num_sym < translations.size(); ++num_sym )
                            front.push( { ng.root, ng.root->dist_2( inv_sym( c0, num_sym ), TF( 0 ) ), int( num_sym ) } );
                    }
                }

                // recursive traversal
                bool interrupted = false;
                while ( ! front.empty() ) {
                    int num_sym = front.top().num_sym;
                    Box *box = front.top().box;
                    front.pop();

                    if ( can_be_evicted( lc, c0, w0, box, num_sym ) )
                        continue;

                    if ( Box* ch = box->last_child ) {
                        do {
                            front.push( { ch, float( ch->dist_2( inv_sym( c0, num_sym ), w0 ) ), num_sym } );
                        } while (( ch = ch->sibling ));
                    } else if ( box->beg_indices < box->end_indices ) {
                        for( TI num_in_ind_1 = box->beg_indices; num_in_ind_1 < box->end_indices; ++num_in_ind_1 ) {
                            TI num_dirac_1 = this->dirac_indices[ num_in_ind_1 ];
                            if ( num_dirac_0 != num_dirac_1 )
                                plane_cut( lc, c0, w0, sym( positions[ num_dirac_1 ], num_sym ), weights[ num_dirac_1 ], num_dirac_1 );
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
                serialized = serialize_rec( positions, weights, { root }, 100000 );
                break;
            }
        }

        // selective send
        std::vector<std::vector<char>> ext;
        mpi->selective_send_and_recv( ext, needs, serialized );

        // update the neighbor info
        for( std::size_t num_in_v_missing_rank = 0; num_in_v_missing_rank < v_missing_ranks.size(); ++num_in_v_missing_rank ) {
            int missing_rank = v_missing_ranks[ num_in_v_missing_rank ];
            if ( Box *box = deserialize_rec( ext[ num_in_v_missing_rank ], missing_rank ) ) {
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

    return err;
}


} // namespace sdot


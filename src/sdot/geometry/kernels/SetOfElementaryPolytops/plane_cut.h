#include <sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/ElementaryPolytopOperations.h>
#include <sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/ShapeCutTmpData.h>
#include <sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/ShapeMap.h>
#include <sdot/geometry/Point.h>

#include <parex/support/StaticRange.h>
#include <parex/TaskRef.h>

using namespace parex;
using namespace sdot;

template<class TI>
void count_to_offsets( TI *count, TI nb_cases, TI nb_lanes ) {
    TI off = 0;
    for( TI num_case = 0; num_case < nb_cases; ++num_case ) {
        for( TI num_lane = 0; num_lane < nb_lanes; ++num_lane ) {
            TI *ptr = count + num_lane * nb_cases + num_case;
            TI val = *ptr;
            *ptr = off;
            off += val;
        }
    }
}

template<class TI,int nb_nodes>
void make_sorted_indices( Vec<TI> &indices, TI *offsets, const TI *cut_cases, TI nb_items, N<nb_nodes> ) {
    constexpr int nb_cases = 1 << nb_nodes;

    SimdRange<SimdSize<TI>::value>::for_each( nb_items, [&]( TI beg_num_item, auto simd_size ) {
        using VI = SimdVec<TI,simd_size.value>;

        VI nc = VI::load_aligned( reinterpret_cast<const TI *>( cut_cases ) + beg_num_item );

        VI io = VI::iota();
        VI oc = io * nb_cases + nc;
        VI of = VI::gather( reinterpret_cast<const TI *>( offsets ), oc );

        VI::scatter( indices.ptr(), of, VI( beg_num_item ) + io );
        VI::scatter( offsets, oc, of + 1 );
    } );
}

template<class TF,class TI,int dim,class A,class B,int nb_nodes>
void make_scalar_products_cut_cases_and_counts( Tensor<TF> &scalar_products, TI *cut_cases, TI *count, const ShapeData<TF,TI,dim> &sd, const A &normals, const B &off_scps, N<nb_nodes> ) {
    constexpr TI nb_cases = 1u << nb_nodes;
    const TI nb_items = sd.ids.size();

    SimdRange<SimdSize<TF>::value>::for_each( nb_items, [&]( TI beg_num_item, auto simd_size ) {
        using VF = SimdVec<TF,simd_size.value>;
        using VI = SimdVec<TI,simd_size.value>;

        // positions
        std::array<std::array<VF,dim>,nb_nodes> pos;
        for( TI n = 0; n < nb_nodes; ++n )
            for( TI d = 0; d < dim; ++d )
                pos[ n ][ d ] = VF::load_aligned( sd.coordinates.ptr( n * dim + d ) + beg_num_item );

        // id
        VI id = VI::load_aligned( sd.ids.data() + beg_num_item );

        // cut info
        std::array<VF,dim> PD;
        for( TI d = 0; d < dim; ++d )
            PD[ d ] = VF::gather( normals.ptr( d ), id );
        VF SD = VF::gather( off_scps.data(), id );

        // scalar products
        std::array<VF,nb_nodes> scp;
        for( TI n = 0; n < nb_nodes; ++n ) {
            scp[ n ] = pos[ n ][ 0 ] * PD[ 0 ];
            for( TI d = 1; d < dim; ++d )
                scp[ n ] = scp[ n ] + pos[ n ][ d ] * PD[ d ];
        }

        // cut case
        VI nc = 0;
        for( TI n = 0; n < nb_nodes; ++n )
            nc = nc + ( as_SimdVec<VI>( scp[ n ] > SD ) & TI( 1 << n ) );

        // store scalar product
        for( TI n = 0; n < nb_nodes; ++n )
            VF::store_aligned( scalar_products.ptr( n ) + beg_num_item, scp[ n ] - SD );

        // store cut case
        VI::store_aligned( cut_cases + beg_num_item, nc );

        // update count
        VI oc = VI::iota() * nb_cases + nc;
        VI::scatter( count, oc, VI::gather( count, oc ) + 1 );
    } );
}

template<class TF,class TI,int dim>
void add_length( Vec<TF> &tmp_scores, const ShapeData<TF,TI,dim> &sd, ShapeCutTmpData<TF,TI> &cm, TI beg_ind, TI end_ind, std::array<TI,4> nn ) {
    // pointer
    std::array<std::array<const TF *,dim>,4> ppos;
    for( TI n = 0; n < 4; ++n )
        for( TI d = 0; d < dim; ++d )
            ppos[ n ][ d ] = sd.coordinates.ptr( nn[ n ] * dim + d );

    std::array<const TF *,4> pscp;
    for( TI n = 0; n < 4; ++n )
        pscp[ n ] = cm.scalar_products.ptr( nn[ n ] );

    for( TI num_ind = beg_ind; num_ind < end_ind; ++num_ind ) {
        TI index = cm.indices[ num_ind ];

        Point<TF,dim> pos_0;
        Point<TF,dim> pos_1;
        Point<TF,dim> pos_2;
        Point<TF,dim> pos_3;
        for( int d = 0; d < dim; ++d ) {
            pos_0[ d ] = ppos[ 0 ][ d ][ index ];
            pos_1[ d ] = ppos[ 1 ][ d ][ index ];
            pos_2[ d ] = ppos[ 2 ][ d ][ index ];
            pos_3[ d ] = ppos[ 3 ][ d ][ index ];
        }

        TF scp_0 = pscp[ 0 ][ index ];
        TF scp_1 = pscp[ 1 ][ index ];
        TF scp_2 = pscp[ 2 ][ index ];
        TF scp_3 = pscp[ 3 ][ index ];

        TF d_0_1 = scp_0 / ( scp_0 - scp_1 );
        TF d_2_3 = scp_2 / ( scp_2 - scp_3 );

        Point<TF,dim> P0 = pos_0 + d_0_1 * ( pos_1 - pos_0 );
        Point<TF,dim> P1 = pos_2 + d_2_3 * ( pos_3 - pos_2 );

        tmp_scores[ num_ind - beg_ind ] += norm_2( P1 - P0 );
    }
}

template<class TF,class TI>
void sort_indices_by_num_sub_case( ShapeCutTmpData<TF,TI> &cm, TI *offsets, TI beg, TI end, const Vec<TI> &num_sub_cases, TI nb_sub_cases ) {
    TI len = end - beg;

    // count scan
    Vec<TI> count( nb_sub_cases + 1, 0 );
    for( TI num_sub_case : num_sub_cases )
        ++count[ num_sub_case ];

    for( TI i = 0, c = 0; i < nb_sub_cases; ++i )
        count[ i ] = std::exchange( c, c + count[ i ] );
    count.back() = num_sub_cases.size();

    // save offsets
    for( TI i = 0; i <= nb_sub_cases; ++i )
        offsets[ i ] = beg + count[ i ];

    // make indices
    Vec<TI> tmp_indices( len );
    for( TI i = 0; i < len; ++i )
        tmp_indices[ count[ num_sub_cases[ i ] ]++ ] = cm.indices[ beg + i ];

    for( TI i = 0; i < len; ++i )
        cm.indices[ beg + i ] = tmp_indices[ i ];
}

template<class TF,class TI,int dim,class A,class B,class C>
ShapeMap<TF,TI,dim> *plane_cut( Task *task, const ShapeMap<TF,TI,dim> &sm, const ElementaryPolytopOperations &eop, const A &normals, const B &scalar_products, const C &/*new_face_ids*/ ) {
    constexpr int max_nb_nodes = 8;

    std::map<std::string,ShapeCutTmpData<TF,TI>> *cdm = new std::map<std::string,ShapeCutTmpData<TF,TI>>; // inp elem => data that will be needed after the first loop
    std::map<std::string,TI> reservation_new_elements; // out elem => nb items to reserve
    for( const auto &p : sm.map ) {
        const ElementaryPolytopOperations::Operations &op = eop.operation_map.find( p.first )->second;
        const ShapeData<TF,TI,dim> &sd = p.second;

        TI nb_nodes = op.nb_nodes;
        TI nb_cases = 1u << nb_nodes;
        TI nb_items = sd.ids.size();

        // tmp data for this shape_type
        ShapeCutTmpData<TF,TI> &cm = (*cdm)[ p.first ];
        cm.scalar_products = Vec<TI>{ nb_items, nb_nodes };

        // scalar_products, cut_cases, count
        Vec<TI> cut_cases( nb_items );
        Vec<TI> count( nb_cases * SimdSize<TF>::value, 0 );
        parex::StaticRange<max_nb_nodes>::for_each( [&]( auto n ) { if ( n.value == nb_nodes )
            make_scalar_products_cut_cases_and_counts( cm.scalar_products, cut_cases.ptr(), count.ptr(), sd, normals, scalar_products, n );
        } );

        // scan of count
        count_to_offsets<TI>( count.ptr(), nb_cases, SimdSize<TF>::value );

        // cut_case_offsets (offset in indices for each case and each subcase)
        const std::vector<ElementaryPolytopOperations::TI> &cut_poss_count = op.cut_info.nb_sub_cases;
        cm.cut_case_offsets.resize( nb_cases ); // for each case, a vector with offsets of each sub case
        for( TI n = 0; n < nb_cases; ++n ) {
            TI dv = n + 1 < nb_cases ? count[ n + 1 ] : nb_items;
            Vec<TI> &cco = cm.cut_case_offsets[ n ];
            cco.resize( cut_poss_count[ n ] + 1, dv );
            cco[ 0 ] = count[ n ];
        }

        //
        cm.indices.resize( nb_items );
        parex::StaticRange<max_nb_nodes>::for_each( [&]( auto n ) { if ( n.value == nb_nodes )
            make_sorted_indices( cm.indices, count.ptr(), cut_cases.ptr(), nb_items, n );
        } );

        // update sub cases
        for( TI num_case = 0; num_case < nb_cases; ++num_case ) {
            if ( cut_poss_count[ num_case ] <= 1 )
                continue;
            TI beg = cm.cut_case_offsets[ num_case ][ 0 ];
            TI end = cm.cut_case_offsets[ num_case ][ 1 ];
            if ( beg == end )
                continue;

            Vec<TF> best_scores( end - beg, 0 );
            for( const std::array<TI,4> &pts : op.cut_info.lengths[ num_case ][ 0 ] )
                add_length( best_scores, sd, cm, beg, end, pts );

            Vec<TI> best_num_sub_case( end - beg, 0 );
            TI nb_sub_cases = op.cut_info.lengths[ num_case ].size();
            for( TI num_sub_case = 1; num_sub_case < nb_sub_cases; ++num_sub_case ) {
                Vec<TF> tmp_scores( end - beg, 0 );
                for( const std::array<TI,4> &pts : op.cut_info.lengths[ num_case ][ num_sub_case ] )
                    add_length( tmp_scores, sd, cm, beg, end, pts );

                for( TI i = 0; i < end - beg; ++i ) {
                    if ( best_scores[ i ] > tmp_scores[ i ] ) {
                        best_num_sub_case[ i ] = num_sub_case;
                        best_scores[ i ] = tmp_scores[ i ];
                    }
                }
            }

            sort_indices_by_num_sub_case( cm, cm.cut_case_offsets[ num_case ].ptr(), beg, end, best_num_sub_case, nb_sub_cases );
        }

        // get reservation for new elements
        for( const auto &p : op.cut_info.nb_output_elements ) {
            auto iter = reservation_new_elements.find( p.first );
            if ( iter == reservation_new_elements.end() )
                iter = reservation_new_elements.insert( iter, { p.first, 0 } );

            for( TI num_case = 0; num_case < nb_cases; ++num_case ) {
                const Vec<TI> &cc = cm.cut_case_offsets[ num_case ];
                for( TI num_sub_case = 0; num_sub_case < cc.size() - 1; ++num_sub_case )
                    iter->second += p.second[ num_case ][ num_sub_case ] * ( cc[ num_sub_case + 1 ] - cc[ num_sub_case + 0 ] );
            }
        }
    }

    // reservation for new elements
    ShapeMap<TF,TI,dim> *res = new ShapeMap<TF,TI,dim>;
    for( const auto &np : reservation_new_elements )
        res->shape_data( np.first, eop, np.second );

    //
    TaskRef task_ref_cdm = Task::ref_on( cdm );
    TaskRef prev_task = task;
    for( const auto &p : sm.map ) {
        const ElementaryPolytopOperations::Operations &op = eop.operation_map.find( p.first )->second;
        const ShapeData<TF,TI,dim> &sd = p.second;
        std::string shape_type = p.first;

        ShapeCutTmpData<TF,TI> &cm = (*cdm)[ shape_type ];

        const std::vector<std::vector<ElementaryPolytopOperations::CutOp>> &cut_ops = op.cut_info.new_elems;
        for( std::size_t num_case = 0; num_case < cut_ops.size(); ++num_case ) {
            for( std::size_t num_sub_case = 0; num_sub_case < cut_ops[ num_case ].size(); ++num_sub_case ) {
                const ElementaryPolytopOperations::CutOp &cut_op = cut_ops[ num_case ][ num_sub_case ];
                TI beg = cm.cut_case_offsets[ num_case ][ num_sub_case + 0 ];
                TI end = cm.cut_case_offsets[ num_case ][ num_sub_case + 1 ];

                if ( end != beg ) {
                    TaskRef new_task = Task::call_r( Kernel::with_task_as_arg( "sdot/geometry/kernels/SetOfElementaryPolytops/mk_items(" + std::to_string( dim ) + " " + cut_op.operation_name + ")" ), {
                        prev_task                                           , // new shape map
                        task->children[ 1 ]                                 , // ElementaryPolytopOperations &eop
                        parex::Task::ref_on( new TI( num_case ) )           ,
                        parex::Task::ref_on( new TI( num_sub_case ) )       ,
                        parex::Task::ref_on( new TI( beg ) )                , // first index
                        parex::Task::ref_on( new TI( end ) )                ,
                        task_ref_cdm                                        , // tmp cut data
                        task->children[ 0 ]                                 , // old shape map
                        parex::Task::ref_on( new std::string( shape_type ) ), // old shape_type
                        task->children[ 3 ]                                   // new_face_ids
                    } );

                    prev_task.task->insert_before_parents( new_task );
                    prev_task = new_task;
                }
            }
        }
    }

    return res;
}

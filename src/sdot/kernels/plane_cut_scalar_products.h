#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>
#include <sstream>
using namespace parex;

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

template<class TI,int nb_cases>
void make_sorted_indices( Vec<TI> &indices, TI *offsets, const TI *cut_cases, TI nb_items, N<nb_cases> ) {
    SimdRange<SimdSize<TI>::value>::for_each( nb_items, [&]( TI beg_num_item, auto simd_size ) {
        using VI = SimdVec<TI,simd_size.value>;

        VI nc = VI::load_aligned( reinterpret_cast<const TI *>( cut_cases ) + beg_num_item );

        VI io = VI::iota();
        VI oc = io * nb_cases + nc;
        VI of = VI::gather( reinterpret_cast<const TI *>( offsets ), oc );

        VI::scatter( reinterpret_cast<TI *>( indices ), of, VI( beg_num_item ) + io );
        VI::scatter( reinterpret_cast<TI *>( offsets ), oc, of + 1 );
    } );
}

template<class TF,class TI,class TN,class VO,class VC,int nb_nodes,int dim>
std::tuple<Vec<Vec<TI>> *,Vec<TI> *,Tensor<TF> *> plane_cut_scalar_products( Tensor<TF> &coordinates, Vec<TI> &ids, TN &normals, VO &off_scps, VC &cut_poss_count, N<nb_nodes>, N<dim> ) {
    const TI nb_items = coordinates.size[ 0 ];
    const TI nb_cases = 1u << nb_nodes;

    // get scalar products, cut cases and counts
    Tensor<TF> *scalar_products = new Tensor<TF>( Vec<TI>{ nb_items, nb_nodes } );
    Vec<TI> count( nb_cases * SimdSize<TF>::value, 0 );
    Vec<TI> cut_cases( nb_items );
    SimdRange<SimdSize<TF>::value>::for_each( nb_items, [&]( TI beg_num_item, auto simd_size ) {
        using VF = SimdVec<TF,simd_size.value>;
        using VI = SimdVec<TI,simd_size.value>;

        // positions
        std::array<std::array<VF,dim>,nb_nodes> pos;
        for( TI n = 0; n < nb_nodes; ++n )
            for( TI d = 0; d < dim; ++d )
                pos[ n ][ d ] = VF::load_aligned( coordinates.ptr( n * dim + d ) + beg_num_item );

        // id
        VI id = VI::load_aligned( ids.data() + beg_num_item );

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
            VF::store_aligned( scalar_products->ptr( n ) + beg_num_item, scp[ n ] - SD );

        // store cut case
        VI::store_aligned( cut_cases.ptr() + beg_num_item, nc );

        // update count
        VI oc = VI::iota() * nb_cases + nc;
        VI::scatter( count.ptr(), oc, VI::gather( count.ptr(), oc ) + 1 );
    } );

    // scan of count
    count_to_offsets<TI>( count.ptr(), nb_cases, SimdSize<TF>::value );

    // cut_case_offsets
    Vec<Vec<TI>> *cut_case_offsets = new Vec<Vec<TI>>( nb_cases );
    for( TI n = 0; n < nb_cases; ++n ) {
        TI dv = n + 1 < nb_cases ? count[ n + 1 ] : nb_items;
        Vec<TI> &cco = cut_case_offsets->operator[]( n );
        cco.resize( cut_poss_count[ n ] + 1, dv );
        cco[ 0 ] = count[ n ];
    }

    //
    Vec<TI> *indices = new Vec<TI>( nb_items );
    make_sorted_indices( *indices, count.ptr(), cut_cases.ptr(), nb_items, N<nb_cases>() );

    return { cut_case_offsets, indices, scalar_products };
}

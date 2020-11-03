#include "../../simd/SimdRange.h"
#include "../../simd/SimdVec.h"
#include "KernelSlot_Cpu.h"

namespace sdot {

template<class TF,class TI,class Arch>
KernelSlot::BI KernelSlot_Cpu<TF,TI,Arch>::nb_lanes_TF() {
    return SimdSize<TF,Arch>::value;
}

template<class TF,class TI,class Arch>
void *KernelSlot_Cpu<TF,TI,Arch>::allocate_TF( BI size ) {
    if ( SimdSize<TF,Arch>::value > 1 )
        return aligned_alloc( SimdSize<TF,Arch>::value * sizeof( TF ), sizeof( TF ) * size );
    return new TF[ size ];
}

template<class TF,class TI,class Arch>
void *KernelSlot_Cpu<TF,TI,Arch>::allocate_TI( BI size ) {
    if ( SimdSize<TI,Arch>::value > 1 )
        return aligned_alloc( SimdSize<TI,Arch>::value * sizeof( TI ), sizeof( TI ) * size );
    return new TI[ size ];
}

template<class TF,class TI,class Arch>
void KernelSlot_Cpu<TF,TI,Arch>::count_to_offsets( void *counts, BI nb_nodes ) {
    TI off = 0;
    BI nb_cases = BI( 1 ) << nb_nodes;
    for( BI num_case = 0; num_case < nb_cases; ++num_case ) {
        for( BI num_lane = 0; num_lane < nb_lanes_TF(); ++num_lane ) {
            TI *ptr = reinterpret_cast<TI *>( counts ) + num_lane * nb_cases + num_case;
            TI val = *ptr;
            *ptr = off;
            off += val;
        }
    }
}

template<class TF,class TI,class Arch>
void KernelSlot_Cpu<TF,TI,Arch>::sort_TI_in_range( BI *out_offsets, void *index_best_sub_case, BI nb_items, BI TI_range, void *aux_TI_ptr, BI aux_TI_off ) {
    const TI *num_case = reinterpret_cast<TI *>( index_best_sub_case );

    // count
    std::vector<TI> offsets( TI_range, 0 );
    for( TI i = 0; i < nb_items; ++i )
        ++offsets[ num_case[ i ] ];

    // scan
    for( TI i = 0, o = 0; i < TI_range; ++i )
        offsets[ i ] = std::exchange( o, o + offsets[ i ] );

    // out_offset
    for( TI i = 0, o = out_offsets[ 0 ]; i < TI_range; ++i )
        out_offsets[ i ] = o + offsets[ i ];

    // write
    TI *aux = reinterpret_cast<TI *>( aux_TI_ptr ) + aux_TI_off;
    std::vector<TI> aux_cp( aux, aux + nb_items );
    for( TI i = 0; i < nb_items; ++i )
        aux[ offsets[ num_case[ i ] ]++ ] = aux_cp[ i ];
}

template<class TF,class TI,class Arch>
void KernelSlot_Cpu<TF,TI,Arch>::sorted_indices( void *indices, void *offsets, const void *cut_cases, BI nb_items, BI nb_nodes ) {
    BI nb_cases = BI( 1 ) << nb_nodes;

    // ...
    SimdRange<SimdSize<TF,Arch>::value>::for_each( nb_items, [&]( TI beg_num_item, auto simd_size ) {
        using VI = SimdVec<TI,simd_size.value,Arch>;

        VI nc = VI::load_aligned( reinterpret_cast<const TI *>( cut_cases ) + beg_num_item );

        VI io = VI::iota();
        VI oc = io * nb_cases + nc;
        VI of = VI::gather( reinterpret_cast<const TI *>( offsets ), oc );

        VI::scatter( reinterpret_cast<TI *>( indices ), of, VI( beg_num_item ) + io );
        VI::scatter( reinterpret_cast<TI *>( offsets ), oc, of + 1 );
    } );

}
template<class TF,class TI,class Arch> template<int dim>
void KernelSlot_Cpu<TF,TI,Arch>::_update_scores( void *score_best_sub_case, void *index_best_sub_case, const ShapeData &sd, BI beg, BI end, BI index_sub_case, const void *num_nodes, BI off_edges, BI len_edges, N<dim> ) {
    using Pt = Point<TF,dim>;

    const TI *inn = reinterpret_cast<const TI *>( num_nodes ) + off_edges;
    const TF *pos = reinterpret_cast<const TF *>( sd.coordinates );

    TF *sv = reinterpret_cast<TF *>( score_best_sub_case );
    TI *iv = reinterpret_cast<TI *>( index_best_sub_case );

    // for each item
    for( TI ind_item = beg; ind_item < end; ++ind_item ) {
        TI num_item = reinterpret_cast<const TI *>( sd.cut_indices )[ ind_item ];

        TF length = 0;
        for( TI num_edge = 0; num_edge < len_edges; ++num_edge ) {
            Pt pts[ 2 ];
            for( TI np = 0; np < 2; ++np ) {
                TI n0 = inn[ 4 * num_edge + 2 * np + 0 ];
                TI n1 = inn[ 4 * num_edge + 2 * np + 1 ];
                TF s0 = reinterpret_cast<const TF *>( sd.cut_out_scps )[ n0 * sd.rese + num_item ];
                TF s1 = reinterpret_cast<const TF *>( sd.cut_out_scps )[ n1 * sd.rese + num_item ];
                TF di = s0 / ( s0 - s1 );
                for( TI d = 0; d < dim; ++d )
                    pts[ np ][ d ] = pos[ ( dim * n0 + d ) * sd.rese + num_item ] +
                              di * ( pos[ ( dim * n1 + d ) * sd.rese + num_item ] -
                                     pos[ ( dim * n0 + d ) * sd.rese + num_item ] );
            }
            length += norm_2( pts[ 1 ] - pts[ 0 ] );
        }


        TF score = 1 / ( length + 1e-40 );
        if ( sv[ ind_item - beg ] < score ) {
            sv[ ind_item - beg ] = score;
            iv[ ind_item - beg ] = index_sub_case;
        }
    }
}

template<class TF,class TI,class Arch> template<int nb_nodes,int dim>
void KernelSlot_Cpu<TF,TI,Arch>::_get_cut_cases( void *cut_cases, void *counts, void *out_sps, const void *coordinates, const void *ids, BI rese, const void **dirs, const void *sps, BI nb_items, N<nb_nodes>, N<dim> ) {
    constexpr BI nb_cases = BI( 1 ) << nb_nodes;

    // get ptrs
    std::array<std::array<const TF *,dim>,nb_nodes> positions;
    for( BI n = 0, i = 0; n < nb_nodes; ++n )
        for( BI d = 0; d < dim; ++d, ++i )
            positions[ n ][ d ] = reinterpret_cast<const TF *>( coordinates ) + i * rese;

    // clear counts
    SimdRange<SimdSize<TI,Arch>::value>::for_each( nb_cases * nb_lanes_TF(), [&]( TI beg_num_item, auto simd_size ) {
        using VI = SimdVec<TI,simd_size.value,Arch>;
        VI::store_aligned( reinterpret_cast<TI *>( counts ) + beg_num_item, TI( 0 ) );
    } );

    // get scalar product, cut cases and counts
    SimdRange<SimdSize<TF,Arch>::value>::for_each( nb_items, [&]( TI beg_num_item, auto simd_size ) {
        using VF = SimdVec<TF,simd_size.value,Arch>;
        using VI = SimdVec<TI,simd_size.value,Arch>;

        // positions
        std::array<std::array<VF,dim>,nb_nodes> pos;
        for( TI n = 0; n < nb_nodes; ++n )
            for( TI d = 0; d < dim; ++d )
                pos[ n ][ d ] = VF::load_aligned( positions[ n ][ d ] + beg_num_item );

        // id
        VI id = VI::load_aligned( reinterpret_cast<const TI *>( ids ) + beg_num_item );

        // cut info
        std::array<VF,dim> PD;
        for( TI d = 0; d < dim; ++d )
            PD[ d ] = VF::gather( reinterpret_cast<const TF *>( dirs[ d ] ), id );
        VF SD = VF::gather( reinterpret_cast<const TF *>( sps ), id );

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
            VF::store_aligned( reinterpret_cast<TF *>( out_sps ) + n * rese + beg_num_item, scp[ n ] - SD );

        // store cut case
        VI::store_aligned( reinterpret_cast<TI *>( cut_cases ) + beg_num_item, nc );

        // update count
        VI oc = VI::iota() * nb_cases + nc;
        VI::scatter( reinterpret_cast<TI *>( counts ), oc, VI::gather( reinterpret_cast<const TI *>( counts ), oc ) + 1 );
    } );
}

template<class TF, class TI,class Arch>
void KernelSlot_Cpu<TF,TI,Arch>::read_TI( BI *dst, const void *src, BI src_off, BI len ) {
    for( BI i = 0; i < len; ++i )
        dst[ i ] = reinterpret_cast<const TI *>( src )[ src_off + i ];
}

template<class TF, class TI,class Arch>
void KernelSlot_Cpu<TF,TI,Arch>::get_local( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size ) {
    std::vector<const BI *> tis_vec;
    std::vector<const double *> tfs_vec;
    _get_local( f, tfs_data, tfs_size, tis_data, tis_size, tfs_vec, tis_vec );
}


template<class TF,class TI,class Arch>
void KernelSlot_Cpu<TF,TI,Arch>::_get_local( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size, std::vector<const double *> &tfs_vec, std::vector<const BI *> &tis_vec ) {
    if ( tfs_size ) {
        if ( local && std::is_same<TF,double>::value ) {
            tfs_vec.push_back( reinterpret_cast<const double *>( std::get<0>( *tfs_data ) ) + std::get<1>( *tfs_data ) );
            return _get_local( f, tfs_data + 1, tfs_size - 1, tis_data, tis_size, tfs_vec, tis_vec );
        }

        std::vector<double> tmp( std::get<2>( *tfs_data ) );
        assign_TF( tmp.data(), 0, std::get<0>( *tfs_data ), std::get<1>( *tfs_data ), std::get<2>( *tfs_data ) );
        tfs_vec.push_back( tmp.data() );
        return _get_local( f, tfs_data + 1, tfs_size - 1, tis_data, tis_size, tfs_vec, tis_vec );
    }
    if ( tis_size ) {
        if ( local && std::is_same<TI,BI>::value ) {
            tis_vec.push_back( reinterpret_cast<const BI *>( std::get<0>( *tis_data ) ) + std::get<1>( *tis_data ) );
            return _get_local( f, tfs_data, tfs_size, tis_data + 1, tis_size - 1, tfs_vec, tis_vec );
        }

        std::vector<BI> tmp( std::get<2>( *tis_data ) );
        assign_TI( tmp.data(), 0, std::get<0>( *tis_data ), std::get<1>( *tis_data ), std::get<2>( *tis_data ) );
        tis_vec.push_back( tmp.data() );
        return _get_local( f, tfs_data, tfs_size, tis_data + 1, tis_size - 1, tfs_vec, tis_vec );
    }

    f( tfs_vec.data(), tis_vec.data() );
}

}

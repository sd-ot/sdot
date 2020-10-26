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

template<class TF,class TI,class Arch> template<int dim>
void KernelSlot_Cpu<TF,TI,Arch>::_get_cut_cases( void *cut_cases, void *offsets, void *out_sps, const void *coordinates, const void *ids, BI rese, const void **dirs, const void *sps, BI nb_items, N<dim> ) {
    constexpr BI nb_nodes = 3;

    std::array<std::array<const TF *,dim>,nb_nodes> positions;
    for( BI n = 0, i = 0; n < nb_nodes; ++n )
        for( BI d = 0; d < dim; ++d, ++i )
            positions[ n ][ d ] = reinterpret_cast<const TF *>( coordinates ) + i * rese;

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

        // num case
        VI nc = 0;
        for( TI n = 0; n < nb_nodes; ++n )
            nc = nc + ( as_SimdVec<VI>( scp[ n ] > SD ) & TI( 1 << n ) );

        // store scalar product
        for( TI n = 0; n < nb_nodes; ++n )
            VF::store_aligned( reinterpret_cast<TF *>( out_sps ) + n * rese + beg_num_item, scp[ n ] - SD );

        // store indices
        for( TI i = 0; i < simd_size.value; ++i )
            reinterpret_cast<TI *>( cut_cases )[ reinterpret_cast<TI *>( offsets )[ nc[ i ] ]++ ] = beg_num_item + i;
    } );
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
            tfs_vec.push_back( reinterpret_cast<const double *>( std::get<0>( *tfs_data ) ) );
            return _get_local( f, tfs_data + 1, tfs_size - 1, tis_data, tis_size, tfs_vec, tis_vec );
        }

        std::vector<double> tmp( std::get<2>( *tfs_data ) );
        assign_TF( tmp.data(), 0, std::get<0>( *tfs_data ), std::get<1>( *tfs_data ), std::get<2>( *tfs_data ) );
        tfs_vec.push_back( tmp.data() );
        return _get_local( f, tfs_data + 1, tfs_size - 1, tis_data, tis_size, tfs_vec, tis_vec );
    }
    if ( tis_size ) {
        if ( local && std::is_same<TI,BI>::value ) {
            tis_vec.push_back( reinterpret_cast<const BI *>( std::get<0>( *tis_data ) ) );
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

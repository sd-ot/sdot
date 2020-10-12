#pragma once

#include "../support/Math.h"
#include "../support/P.h"
#include "KernelSlot.h"
#include <numeric>

namespace sdot {

/**
*/
template<class TF,class TI>
class Kernels_Cpu_Gen : public KernelSlot {
public:
    enum {           local                     = true };

    /**/             Kernels_Cpu_Gen           ( std::string slot ) : KernelSlot( slot ) {}

    virtual void     assign_repeated_TF        ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TF *>( dst ) + dst_off, reinterpret_cast<TF *>( dst ) + dst_off + len, reinterpret_cast<const TF *>( src )[ src_off ] ); }
    virtual void     assign_repeated_TI        ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TI *>( dst ) + dst_off, reinterpret_cast<TI *>( dst ) + dst_off + len, reinterpret_cast<const TI *>( src )[ src_off ] ); }
    virtual void     write_to_stream           ( std::ostream &os ) const { os << "Kernel(" << slot_name << ",Gen)"; }
    virtual void     assign_iota_TI            ( void *dst, BI dst_off, BI src_off, BI len ) override { std::iota( reinterpret_cast<TI *>( dst ) + dst_off, reinterpret_cast<TI *>( dst ) + dst_off + len, src_off ); }
    virtual BI       nb_multiprocs             () override { return 1; }
    virtual void*    allocate_TF               ( BI size ) override { return new TF[ size ]; }
    virtual void*    allocate_TI               ( BI size ) override { return new TI[ size ]; }
    virtual void     display_TF                ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TF *>( ptr )[ off + i ]; }
    virtual void     display_TI                ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TI *>( ptr )[ off + i ]; }
    virtual void     assign_TF                 ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = reinterpret_cast<const TF *>( src )[ n + src_off ]; }
    virtual void     assign_TI                 ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = reinterpret_cast<const TI *>( src )[ n + src_off ]; }
    virtual void     get_local                 ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size ) override;
    virtual void     free_TF                   ( void *ptr ) override { delete [] reinterpret_cast<TF *>( ptr ); }
    virtual void     free_TI                   ( void *ptr ) override { delete [] reinterpret_cast<TI *>( ptr ); }
    virtual double   score                     () { return 0; }

    #define          POSSIBLE_TF( T )          \
      virtual void   assign_TF                 ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = src[ n + src_off ]; } \
      virtual void   assign_TF                 ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) dst[ n + dst_off ] = reinterpret_cast<const TF *>( src )[ n + src_off ]; }
    #include         "possible_TFs.h"
    #undef           POSSIBLE_TF

    #define          POSSIBLE_TI( T )          \
      virtual void   assign_TI                 ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = src[ n + src_off ]; } \
      virtual void   assign_TI                 ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) dst[ n + dst_off ] = reinterpret_cast<const TI *>( src )[ n + src_off ]; }
    #include         "possible_TIs.h"
    #undef           POSSIBLE_TI

    virtual BI       init_offsets_for_cut_cases( void *off_0, void *off_1, BI nb_nodes, BI nb_items ) override;
    #define          POSSIBLE_DIM( DIM )       \
    virtual void     get_cut_cases             ( void *cut_cases, void *offsets, const void *coordinates, const void *ids, const void **normals, const void *scalar_products, BI nb_items, N<DIM> nd ) override { _get_cut_cases( cut_cases, offsets, coordinates, ids, normals, scalar_products, nb_items, nd ); }
    #include         "possible_DIMs.h"
    #undef           POSSIBLE_DIM

private:
    template         <int dim>
    void             _get_cut_cases            ( void *cut_cases, void *offsets, const void *coordinates, const void *ids, const void **normals, const void *scalar_products, BI nb_items, N<dim> );
    void             _get_local                ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size, std::vector<const double *> &tfs_vec, std::vector<const BI *> &tis_vec );
};

// =============================================================================================================================
template<class TF,class TI>
typename Kernels_Cpu_Gen<TF,TI>::BI Kernels_Cpu_Gen<TF,TI>::init_offsets_for_cut_cases( void *off_0, void *off_1, BI nb_nodes, BI nb_items ) {
    // off_x
    BI nb_cases = 1u << nb_nodes;
    BI re_items = ceil( nb_items, 1 );
    for( BI n = 0, i = 0; n < nb_cases; ++n ) {
        for( BI m = 0; m < nb_multiprocs(); ++m, ++i ) {
            BI off = re_items * n + m * nb_items / nb_multiprocs();
            reinterpret_cast<TI *>( off_0 )[ i ] = off;
            reinterpret_cast<TI *>( off_1 )[ i ] = off;
        }
    }

    //
    return re_items * nb_cases;
}

template<class TF,class TI> template<int dim>
void Kernels_Cpu_Gen<TF,TI>::_get_cut_cases( void *cut_cases, void *offsets, const void *coordinates, const void *ids, const void **normals, const void *scalar_products, BI nb_items, N<dim> ) {
    for( BI n = 0; n < nb_items; ++n ) {
        //

        //
    }
}

template<class TF, class TI>
void Kernels_Cpu_Gen<TF,TI>::get_local( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size ) {
    std::vector<const BI *> tis_vec;
    std::vector<const double *> tfs_vec;
    _get_local( f, tfs_data, tfs_size, tis_data, tis_size, tfs_vec, tis_vec );
}

template<class TF,class TI>
void Kernels_Cpu_Gen<TF,TI>::_get_local( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size, std::vector<const double *> &tfs_vec, std::vector<const BI *> &tis_vec ) {
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

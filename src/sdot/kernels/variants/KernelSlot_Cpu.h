#ifndef SDOT_KernelSlot_Cpu_HEADER
#define SDOT_KernelSlot_Cpu_HEADER

#include "../../support/TODO.h"
#include "../../support/Math.h"
#include "../../support/P.h"

#include "../../geometry/ShapeData.h"

#include "../KernelSlot.h"

#include <numeric>

namespace sdot {

/**
*/
template<class TF,class TI,class Arch>
class KernelSlot_Cpu : public KernelSlot {
public:
    enum {           local                    = true };

    /**/             KernelSlot_Cpu           ( std::string slot, double score ) : KernelSlot( slot ), _score( score ) {}

    virtual void     assign_repeated_TF       ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TF *>( dst ) + dst_off, reinterpret_cast<TF *>( dst ) + dst_off + len, reinterpret_cast<const TF *>( src )[ src_off ] ); }
    virtual void     assign_repeated_TI       ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TI *>( dst ) + dst_off, reinterpret_cast<TI *>( dst ) + dst_off + len, reinterpret_cast<const TI *>( src )[ src_off ] ); }
    virtual void     assign_iota_TI           ( void *dst, BI dst_off, BI src_off, BI len ) override { std::iota( reinterpret_cast<TI *>( dst ) + dst_off, reinterpret_cast<TI *>( dst ) + dst_off + len, src_off ); }
    virtual void     assign_TF                ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = reinterpret_cast<const TF *>( src )[ n + src_off ]; }
    virtual void     assign_TI                ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = reinterpret_cast<const TI *>( src )[ n + src_off ]; }
    virtual void     get_local                ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size ) override;
    virtual void     read_TI                  ( BI *dst, const void *src, BI src_off, BI len );

    virtual void     write_to_stream          ( std::ostream &os ) const { os << "Kernel(" << slot_name << "," << Arch::name() << ")"; }
    virtual void     display_TF               ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TF *>( ptr )[ off + i ]; }
    virtual void     display_TI               ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TI *>( ptr )[ off + i ]; }

    virtual BI       nb_multiprocs            () override { return 1; }
    virtual BI       nb_lanes_TF              () override;

    virtual void*    allocate_TF              ( BI size ) override;
    virtual void*    allocate_TI              ( BI size ) override;
    virtual void     free_TF                  ( void *ptr ) override { free( reinterpret_cast<TF *>( ptr ) ); }
    virtual void     free_TI                  ( void *ptr ) override { free( reinterpret_cast<TI *>( ptr ) ); }

    virtual double   score                    () { return _score; }

    #define          POSSIBLE_TF( T )         \
    virtual void     assign_TF                ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = src[ n + src_off ]; } \
    virtual void     assign_TF                ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) dst[ n + dst_off ] = reinterpret_cast<const TF *>( src )[ n + src_off ]; }
    #include         "../possible_TFs.h"
    #undef           POSSIBLE_TF

    #define          POSSIBLE_TI( T )         \
    virtual void     assign_TI                ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = src[ n + src_off ]; } \
    virtual void     assign_TI                ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) dst[ n + dst_off ] = reinterpret_cast<const TI *>( src )[ n + src_off ]; }
    #include         "../possible_TIs.h"
    #undef           POSSIBLE_TI

    #define          POSSIBLE_NB_NODES_AND_DIM( NB_NODES, DIM ) \
    virtual void     get_cut_cases            ( void *cut_cases, void *offsets, void *out_sps, const void *coordinates, const void *ids, BI rese, const void **normals, const void *scalar_products, BI nb_items, N<NB_NODES> nn, N<DIM> nd ) override { _get_cut_cases( cut_cases, offsets, out_sps, coordinates, ids, rese, normals, scalar_products, nb_items, nn, nd ); }
    #include         "../possible_NB_NODES_AND_DIMs.h"
    #undef           POSSIBLE_NB_NODES_AND_DIM

    virtual void     count_to_offsets         ( void *counts, BI nb_nodes );
    virtual void     sorted_indices           ( void *indices, void *offsets, const void *cut_cases, BI nb_items, BI nb_nodes );

    #include         "KernelSlot_gen_def_cpu.h"

private:
    template         <int nb_nodes,int dim>
    void             _get_cut_cases           ( void *cut_cases, void *counts, void *out_sps, const void *coordinates, const void *ids, BI rese, const void **normals, const void *scalar_products, BI nb_items, N<nb_nodes>, N<dim> );
    void             _get_local               ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size, std::vector<const double *> &tfs_vec, std::vector<const BI *> &tis_vec );

    double           _score;
};

}

#include "KernelSlot_Cpu.tcc"

#endif // SDOT_KernelSlot_Cpu_HEADER

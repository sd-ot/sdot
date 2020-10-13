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
    enum {           local                     = true };

    /**/             KernelSlot_Cpu            ( std::string slot, double score ) : KernelSlot( slot ), _score( score ) {}

    virtual void     assign_repeated_TF        ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TF *>( dst ) + dst_off, reinterpret_cast<TF *>( dst ) + dst_off + len, reinterpret_cast<const TF *>( src )[ src_off ] ); }
    virtual void     assign_repeated_TI        ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TI *>( dst ) + dst_off, reinterpret_cast<TI *>( dst ) + dst_off + len, reinterpret_cast<const TI *>( src )[ src_off ] ); }
    virtual void     write_to_stream           ( std::ostream &os ) const { os << "Kernel(" << slot_name << "," << Arch::name() << ")"; }
    virtual void     assign_iota_TI            ( void *dst, BI dst_off, BI src_off, BI len ) override { std::iota( reinterpret_cast<TI *>( dst ) + dst_off, reinterpret_cast<TI *>( dst ) + dst_off + len, src_off ); }
    virtual BI       nb_multiprocs             () override { return 1; }
    virtual void*    allocate_TF               ( BI size ) override;
    virtual void*    allocate_TI               ( BI size ) override;
    virtual void     display_TF                ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TF *>( ptr )[ off + i ]; }
    virtual void     display_TI                ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TI *>( ptr )[ off + i ]; }
    virtual void     assign_TF                 ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = reinterpret_cast<const TF *>( src )[ n + src_off ]; }
    virtual void     assign_TI                 ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = reinterpret_cast<const TI *>( src )[ n + src_off ]; }
    virtual void     get_local                 ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size ) override;
    virtual void     free_TF                   ( void *ptr ) override { delete [] reinterpret_cast<TF *>( ptr ); }
    virtual void     free_TI                   ( void *ptr ) override { delete [] reinterpret_cast<TI *>( ptr ); }
    virtual double   score                     () { return _score; }

    #define          POSSIBLE_TF( T )          \
      virtual void   assign_TF                 ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = src[ n + src_off ]; } \
      virtual void   assign_TF                 ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) dst[ n + dst_off ] = reinterpret_cast<const TF *>( src )[ n + src_off ]; }
    #include         "../possible_TFs.h"
    #undef           POSSIBLE_TF

    #define          POSSIBLE_TI( T )          \
      virtual void   assign_TI                 ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = src[ n + src_off ]; } \
      virtual void   assign_TI                 ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) dst[ n + dst_off ] = reinterpret_cast<const TI *>( src )[ n + src_off ]; }
    #include         "../possible_TIs.h"
    #undef           POSSIBLE_TI

    virtual BI       init_offsets_for_cut_cases( void *off_0, void *off_1, BI nb_nodes, BI nb_items ) override;

    #define          POSSIBLE_DIM( DIM )       \
    virtual void     get_cut_cases             ( void *cut_cases, void *offsets, void *out_sps, const void *coordinates, const void *ids, BI rese, const void **normals, const void *scalar_products, BI nb_items, N<DIM> nd ) override { _get_cut_cases( cut_cases, offsets, out_sps, coordinates, ids, rese, normals, scalar_products, nb_items, nd ); }
    #include         "../possible_DIMs.h"
    #undef           POSSIBLE_DIM

    virtual void     mk_items_0_0_1_1_2_2      ( ShapeData &new_shape_data, const std::array<BI,3> &new_node_indices, const ShapeData &old_shape_data, const std::array<BI,3> &old_node_indices, BI num_case, const void */*cut_ids*/, N<2> dim ) override {
        TF *new_x_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 0 ) * new_shape_data.rese;
        TF *new_y_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 1 ) * new_shape_data.rese;
        TF *new_x_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 0 ) * new_shape_data.rese;
        TF *new_y_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 1 ) * new_shape_data.rese;
        TF *new_x_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 0 ) * new_shape_data.rese;
        TF *new_y_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 1 ) * new_shape_data.rese;

        TI *new_f_0 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 0 ] * new_shape_data.rese;
        TI *new_f_1 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 1 ] * new_shape_data.rese;
        TI *new_f_2 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 2 ] * new_shape_data.rese;

        TI *new_ids = reinterpret_cast<TI *>( new_shape_data.ids );

        const TF *old_x_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 0 ) * old_shape_data.rese;
        const TF *old_y_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 1 ) * old_shape_data.rese;
        const TF *old_x_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 0 ) * old_shape_data.rese;
        const TF *old_y_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 1 ) * old_shape_data.rese;
        const TF *old_x_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 0 ) * old_shape_data.rese;
        const TF *old_y_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 1 ) * old_shape_data.rese;

        const TI *old_f_0 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 0 ] * old_shape_data.rese;
        const TI *old_f_1 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 1 ] * old_shape_data.rese;
        const TI *old_f_2 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 2 ] * old_shape_data.rese;

        const TI *old_ids = reinterpret_cast<const TI *>( old_shape_data.ids );

        const TI *o0 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_0 ] );
        const TI *o1 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_1 ] );
        const TI *cc = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::cut_case ] );

        for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
            for( BI ind = o0[ num_case * nb_multiprocs() + nmp ]; ind < o1[ num_case * nb_multiprocs() + nmp ]; ++ind ) {
                TI off = cc[ ind ];

                new_x_0[ new_shape_data.size ] = old_x_0[ off ];
                new_y_0[ new_shape_data.size ] = old_y_0[ off ];
                new_x_1[ new_shape_data.size ] = old_x_1[ off ];
                new_y_1[ new_shape_data.size ] = old_y_1[ off ];
                new_x_2[ new_shape_data.size ] = old_x_2[ off ];
                new_y_2[ new_shape_data.size ] = old_y_2[ off ];

                new_f_0[ new_shape_data.size ] = old_f_0[ off ];
                new_f_1[ new_shape_data.size ] = old_f_1[ off ];
                new_f_2[ new_shape_data.size ] = old_f_2[ off ];

                new_ids[ new_shape_data.size ] = old_ids[ off ];

                ++new_shape_data.size;
            }
        }
    }

private:
    template         <int dim>
    void             _get_cut_cases            ( void *cut_cases, void *offsets, void *out_sps, const void *coordinates, const void *ids, BI rese, const void **normals, const void *scalar_products, BI nb_items, N<dim> );
    void             _get_local                ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size, std::vector<const double *> &tfs_vec, std::vector<const BI *> &tis_vec );

    double           _score;
};

}

#include "KernelSlot_Cpu.tcc"

#endif // SDOT_KernelSlot_Cpu_HEADER

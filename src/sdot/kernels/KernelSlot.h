#pragma once

#include "../support/TypeName.h"
#include "../support/N.h"
#include "../support/S.h"
#include <functional>
#include <ostream>
#include <memory>
#include <vector>
#include <string>

namespace sdot {
class ShapeData;

/**
*/
class KernelSlot {
public:
    using            VK                        = std::vector<std::unique_ptr<KernelSlot>>;
    using            BI                        = std::uint64_t;

    /**/             KernelSlot                ( std::string slot_name );
    virtual         ~KernelSlot                ();

    virtual void     assign_repeated_TF        ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void     assign_repeated_TI        ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void     write_to_stream           ( std::ostream &os ) const = 0;
    virtual void     assign_iota_TI            ( void *dst, BI dst_off, BI src_off, BI len ) = 0;
    virtual BI       nb_multiprocs             () = 0;
    virtual void*    allocate_TF               ( BI size ) = 0;
    virtual void*    allocate_TI               ( BI size ) = 0;
    virtual void     display_TF                ( std::ostream &os, const void *data, BI off, BI len ) = 0;
    virtual void     display_TI                ( std::ostream &os, const void *data, BI off, BI len ) = 0;
    virtual void     get_local                 ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size ) = 0;
    virtual void     assign_TF                 ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void     assign_TI                 ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void     free_TF                   ( void *ptr ) = 0;
    virtual void     free_TI                   ( void *ptr ) = 0;
    virtual double   score                     () = 0; ///<

    #define          POSSIBLE_TF( T )          \
      virtual void   assign_TF                 ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) = 0; \
      virtual void   assign_TF                 ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    #include         "possible_TFs.h"
    #undef           POSSIBLE_TF

    #define          POSSIBLE_TI( T )          \
      virtual void   assign_TI                 ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) = 0; \
      virtual void   assign_TI                 ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    #include         "possible_TIs.h"
    #undef           POSSIBLE_TI

    virtual BI       init_offsets_for_cut_cases( void *off_0, void *off_1, BI nb_cases, BI nb_items ) = 0;

    #define          POSSIBLE_DIM( DIM )       \
      virtual void   get_cut_cases             ( void *cut_cases, void *offsets, void *out_sps, const void *coordinates, const void *ids, BI rese, const void **normals, const void *scalar_products, BI nb_items, N<DIM> nd ) = 0;
    #include         "possible_DIMs.h"
    #undef           POSSIBLE_DIM

    virtual void     mk_items_0_0_1_1_2_2      ( ShapeData &new_shape_data, const std::array<BI,3> &new_node_indices, const ShapeData &old_shape_data, const std::array<BI,3> &old_node_indices, BI num_case, BI cut_id, N<2> ) = 0;

    template         <class TF,class TI>
    static  VK       available_slots           ( S<TF>, S<TI> ) { return available_slots( TypeName<TF>::name(), TypeName<TI>::name() ); }
    static  VK       available_slots           ( std::string TF = TypeName<double>::name(), std::string TI = TypeName<std::uint64_t>::name() );

    std::string      slot_name;
};

} // namespace sdot

#pragma once

#include "../support/TypeName.h"
#include "../support/S.h"
#include <functional>
#include <ostream>
#include <memory>
#include <vector>
#include <string>

namespace sdot {

/**
*/
class KernelSlot {
public:
    using           VK                = std::vector<std::unique_ptr<KernelSlot>>;
    using           BI                = std::uint64_t;

    /**/            KernelSlot        ( std::string slot_name );

    virtual void    assign_repeated_TF( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void    assign_repeated_TI( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void    write_to_stream   ( std::ostream &os ) const = 0;
    virtual void*   allocate_TF       ( BI size ) = 0;
    virtual void*   allocate_TI       ( BI size ) = 0;
    virtual void    display_TF        ( std::ostream &os, const void *data, BI off, BI len ) = 0;
    virtual void    display_TI        ( std::ostream &os, const void *data, BI off, BI len ) = 0;
    virtual void    get_local         ( const std::function<void( const double **tfs, const BI **tis )> &f, const std::tuple<const void *,BI,BI> *tfs_data, BI tfs_size, const std::tuple<const void *,BI,BI> *tis_data, BI tis_size ) = 0;
    virtual void    assign_TF         ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void    assign_TI         ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    virtual void    free_TF           ( void *ptr ) = 0;
    virtual void    free_TI           ( void *ptr ) = 0;
    virtual double  score             () = 0; ///<

    #define POSSIBLE_TF( T ) \
      virtual void  assign_TF         ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) = 0; \
      virtual void  assign_TF         ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    #include "possible_TFs.h"
    #undef POSSIBLE_TF

    #define POSSIBLE_TI( T ) \
      virtual void  assign_TI         ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) = 0; \
      virtual void  assign_TI         ( T *dst, BI dst_off, const void *src, BI src_off, BI len ) = 0;
    #include "possible_TIs.h"
    #undef POSSIBLE_TI

    template        <class TF,class TI>
    static  VK      available_slots   ( S<TF>, S<TI> ) { return available_slots( TypeName<TF>::name(), TypeName<TI>::name() ); }
    static  VK      available_slots   ( std::string TF = TypeName<double>::name(), std::string TI = TypeName<std::uint64_t>::name() );

    std::string     slot_name;
};

} // namespace sdot

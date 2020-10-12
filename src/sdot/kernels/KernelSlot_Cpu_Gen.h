#pragma once

#include "KernelSlot.h"

namespace sdot {

/**
*/
template<class TF,class TI>
class Kernels_Cpu_Gen : public KernelSlot {
public:
    /**/           Kernels_Cpu_Gen   ( std::string slot ) : KernelSlot( slot ) {}

    virtual void   assign_repeated_TF( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TF *>( dst ) + dst_off, reinterpret_cast<TF *>( dst ) + dst_off + len, reinterpret_cast<const TF *>( src )[ src_off ] ); }
    virtual void   assign_repeated_TI( void *dst, BI dst_off, const void *src, BI src_off, BI len ) override { std::fill( reinterpret_cast<TI *>( dst ) + dst_off, reinterpret_cast<TI *>( dst ) + dst_off + len, reinterpret_cast<const TI *>( src )[ src_off ] ); }
    virtual void   write_to_stream   ( std::ostream &os ) const { os << "Kernel(" << slot_name << ",Gen)"; }
    virtual void*  allocate_TF       ( BI size ) override { return new TF[ size ]; }
    virtual void*  allocate_TI       ( BI size ) override { return new TI[ size ]; }
    virtual void   display_TF        ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TF *>( ptr )[ off + i ]; }
    virtual void   display_TI        ( std::ostream &os, const void *ptr, BI off, BI len ) { for( TI i = 0; i < len; ++i ) os << ( i ? "," : "" ) << reinterpret_cast<const TI *>( ptr )[ off + i ]; }
    virtual void   assign_TF         ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = reinterpret_cast<const TF *>( src )[ n + src_off ]; }
    virtual void   assign_TI         ( void *dst, BI dst_off, const void *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = reinterpret_cast<const TI *>( src )[ n + src_off ]; }
    virtual void   free_TF           ( void *ptr ) override { delete [] reinterpret_cast<TF *>( ptr ); }
    virtual void   free_TI           ( void *ptr ) override { delete [] reinterpret_cast<TI *>( ptr ); }
    virtual double score             () { return 0; }

    #define POSSIBLE_TF( T ) \
      virtual void assign_TF         ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TF *>( dst )[ n + dst_off ] = src[ n + src_off ]; }
    #include "possible_TFs.h"
    #undef POSSIBLE_TF

    #define POSSIBLE_TI( T ) \
      virtual void assign_TI         ( void *dst, BI dst_off, const T *src, BI src_off, BI len ) { for( TI n = 0; n < len; ++n ) reinterpret_cast<TI *>( dst )[ n + dst_off ] = src[ n + src_off ]; }
    #include "possible_TIs.h"
    #undef POSSIBLE_TI
};

}

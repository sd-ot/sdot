#pragma once

#include "KernelSlot.h"

namespace sdot {

/**
*/
class VecTT {
public:
    using       BI             = KernelSlot::BI;

    template    <class TV>
    /**/        VecTT          ( KernelSlot *ks, const std::vector<TV> &values );
    /**/        VecTT          ( KernelSlot *ks, BI rese, BI size );
    /**/        VecTT          ( KernelSlot *ks, BI size = 0 );
    /**/        VecTT          ( const VecTT &that );
    /**/        VecTT          ( VecTT &&that );

    /**/       ~VecTT          ();

    VecTT&      operator=      ( const VecTT &that );
    VecTT&      operator=      ( VecTT &&that );

    void        write_to_stream( std::ostream &os ) const;
    void        display        ( std::ostream &os, BI off, BI len ) const;
    BI          size           () const;

    const void* data           () const;
    void*       data           ();

    void        reserve        ( BI new_size, bool copy_if_resize = true );
    void        resize         ( BI new_size, bool copy_if_resize = true );
    void        free           ();

private:
    BI          _rese;
    BI          _size;
    void*       _data;
    KernelSlot* ks;
};

// ======================================================================
template<class TV>
VecTT::VecTT( KernelSlot *ks, const std::vector<TV> &values ) : VecTT( ks, values.size(), values.size() ) {
    ks->assign_TT( _data, 0, values.data(), 0, values.size() );
}

} // namespace sdot

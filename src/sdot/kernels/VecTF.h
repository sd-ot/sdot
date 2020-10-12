#pragma once

#include "KernelSlot.h"

namespace sdot {

/**
*/
class VecTF {
public:
    using       BI             = KernelSlot::BI;

    template    <class TV>
    /**/        VecTF          ( KernelSlot *ks, const std::vector<TV> &values );
    /**/        VecTF          ( KernelSlot *ks, BI rese, BI size );
    /**/        VecTF          ( KernelSlot *ks, BI size = 0 );
    /**/        VecTF          ( const VecTF &that );
    /**/        VecTF          ( VecTF &&that );

    /**/       ~VecTF          ();

    VecTF&      operator=      ( const VecTF &that );
    VecTF&      operator=      ( VecTF &&that );

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
VecTF::VecTF( KernelSlot *ks, const std::vector<TV> &values ) : VecTF( ks, values.size(), values.size() ) {
    ks->assign_TF( _data, 0, values.data(), 0, values.size() );
}

} // namespace sdot

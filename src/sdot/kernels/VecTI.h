#pragma once

#include "KernelSlot.h"

namespace sdot {

/**
*/
class VecTI {
public:
    using       BI             = KernelSlot::BI;

    template    <class TV>
    /**/        VecTI          ( KernelSlot *ks, const std::vector<TV> &values );
    /**/        VecTI          ( KernelSlot *ks, BI rese, BI size );
    /**/        VecTI          ( KernelSlot *ks, BI size = 0 );
    /**/        VecTI          ( const VecTI &that );
    /**/        VecTI          ( VecTI &&that );

    /**/       ~VecTI          ();

    VecTI&      operator=      ( const VecTI &that );
    VecTI&      operator=      ( VecTI &&that );

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
VecTI::VecTI( KernelSlot *ks, const std::vector<TV> &values ) : VecTI( ks, values.size(), values.size() ) {
    ks->assign_TI( _data, 0, values.data(), 0, values.size() );
}

} // namespace sdot

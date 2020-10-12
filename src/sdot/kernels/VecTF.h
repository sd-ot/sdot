#pragma once

#include "KernelSlot.h"

namespace sdot {

/**
*/
class VecTF {
public:
    using       BI             = KernelSlot::BI;

    template    <class TF>
    /**/        VecTF          ( KernelSlot *ks, const std::vector<TF> &values );
    /**/        VecTF          ( KernelSlot *ks, BI rese = 0, BI size = 0 );
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

    void        resize         ( BI new_size );
    void        free           ();

private:
    BI          _rese;
    BI          _size;
    void*       _data;
    KernelSlot *ks;
};

// ======================================================================
template<class TF>
VecTF::VecTF( KernelSlot *ks, const std::vector<TF> &values ) : VecTF( ks, values.size(), values.size() ) {
    ks->assign_TF( _data, 0, values.data(), 0, values.size() );
}

} // namespace sdot

#include "../support/TODO.h"
#include "VecTF.h"

namespace sdot {

VecTF::VecTF( KernelSlot *ks, BI rese, BI size ) : ks( ks ) {
    _data = ks->allocate_TF( rese );
    _rese = rese;
    _size = size;
}

VecTF::VecTF( const VecTF &that ) : VecTF( that.ks, that.size() ) {
    ks->assign_TF( _data, 0, that._data, 0, _size );
}

VecTF::VecTF( VecTF &&that ) : ks( that.ks ) {
    _data = std::exchange( that._data, nullptr );
    _size = std::exchange( that._size, 0 );
}

VecTF &VecTF::operator=( const VecTF &that ) {
    resize( that.size() );
    ks->assign_TF( _data, 0, that._data, 0, _size );
    return *this;
}

VecTF &VecTF::operator=( VecTF &&that ) {
    free();

    _data = std::exchange( that._data, nullptr );
    _size = std::exchange( that._size, 0 );
    ks = that.ks;

    return *this;
}

void VecTF::write_to_stream( std::ostream &os ) const {
    ks->display_TF( os, _data, 0, _size );
}

void VecTF::display( std::ostream &os, BI off, BI len ) const {
    ks->display_TF( os, data(), off, len );
}

VecTF::BI VecTF::size() const {
    return _size;
}

const void *VecTF::data() const {
    return _data;
}

void *VecTF::data() {
    return _data;
}

void VecTF::resize( BI new_size ) {
    TODO;
}

void VecTF::free() {
    if ( _size ) {
        ks->free_TF( _data );
        _size = 0;
    }
}

VecTF::~VecTF() {
    free();
}

} // namespace sdot

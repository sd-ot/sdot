#include "../support/TODO.h"
#include "VecTF.h"

namespace sdot {

VecTF::VecTF( KernelSlot *ks, BI rese, BI size ) : ks( ks ) {
    if ( rese )
        _data = ks->allocate_TF( rese );
    _rese = rese;
    _size = size;
}

VecTF::VecTF( KernelSlot *ks, BI size ) : VecTF( ks, size, size ) {
}

VecTF::VecTF( const VecTF &that ) : VecTF( that.ks, that.size(), that.size() ) {
    ks->assign_TF( _data, 0, that._data, 0, _size );
}

VecTF::VecTF( VecTF &&that ) : ks( that.ks ) {
    _data = std::exchange( that._data, nullptr );
    _rese = std::exchange( that._rese, 0 );
    _size = std::exchange( that._size, 0 );
}

VecTF::~VecTF() {
    free();
}

VecTF &VecTF::operator=( const VecTF &that ) {
    resize( that.size() );
    ks->assign_TF( _data, 0, that._data, 0, _size );
    return *this;
}

VecTF &VecTF::operator=( VecTF &&that ) {
    free();

    _data = std::exchange( that._data, nullptr );
    _rese = std::exchange( that._rese, 0 );
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

void VecTF::reserve( BI new_size, bool copy_if_resize ) {
    if ( _rese >= new_size )
        return;

    void *old_data = _data;
    BI old_rese = _rese;

    // update _rese
    if ( ! _rese )
        _rese = 1;
    while ( _rese < new_size )
        _rese *= 2;

    // update _data
    _data = ks->allocate_TF( _rese );
    if ( copy_if_resize )
        ks->assign_TF( _data, 0, old_data, 0, _size );
    if ( old_rese )
        ks->free_TF( old_data );
}

void VecTF::resize( BI new_size, bool copy_if_resize ) {
    reserve( new_size, copy_if_resize );
    _size = new_size;
}

void VecTF::free() {
    if ( _rese ) {
        ks->free_TF( _data );
        _rese = 0;
        _size = 0;
    }
}

} // namespace sdot

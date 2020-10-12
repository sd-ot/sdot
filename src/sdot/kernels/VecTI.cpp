#include "../support/TODO.h"
#include "VecTI.h"

namespace sdot {

VecTI::VecTI( KernelSlot *ks, BI rese, BI size ) : ks( ks ) {
    if ( rese )
        _data = ks->allocate_TI( rese );
    _rese = rese;
    _size = size;
}

VecTI::VecTI( KernelSlot *ks, BI size ) : VecTI( ks, size, size ) {
}

VecTI::VecTI( const VecTI &that ) : VecTI( that.ks, that.size(), that.size() ) {
    ks->assign_TI( _data, 0, that._data, 0, _size );
}

VecTI::VecTI( VecTI &&that ) : ks( that.ks ) {
    _data = std::exchange( that._data, nullptr );
    _rese = std::exchange( that._rese, 0 );
    _size = std::exchange( that._size, 0 );
}

VecTI::~VecTI() {
    free();
}

VecTI &VecTI::operator=( const VecTI &that ) {
    resize( that.size() );
    ks->assign_TI( _data, 0, that._data, 0, _size );
    return *this;
}

VecTI &VecTI::operator=( VecTI &&that ) {
    free();

    _data = std::exchange( that._data, nullptr );
    _rese = std::exchange( that._rese, 0 );
    _size = std::exchange( that._size, 0 );
    ks = that.ks;

    return *this;
}

void VecTI::write_to_stream( std::ostream &os ) const {
    ks->display_TI( os, _data, 0, _size );
}

void VecTI::display( std::ostream &os, BI off, BI len ) const {
    ks->display_TI( os, data(), off, len );
}

VecTI::BI VecTI::size() const {
    return _size;
}

const void *VecTI::data() const {
    return _data;
}

void *VecTI::data() {
    return _data;
}

void VecTI::reserve( BI new_size, bool copy_if_resize ) {
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
    _data = ks->allocate_TI( _rese );
    if ( copy_if_resize )
        ks->assign_TI( _data, 0, old_data, 0, _size );
    if ( old_rese )
        ks->free_TI( old_data );
}

void VecTI::resize( BI new_size, bool copy_if_resize ) {
    reserve( new_size, copy_if_resize );
    _size = new_size;
}

void VecTI::free() {
    if ( _rese ) {
        ks->free_TI( _data );
        _rese = 0;
        _size = 0;
    }
}

} // namespace sdot

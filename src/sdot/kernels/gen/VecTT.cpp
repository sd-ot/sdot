#include "../support/TODO.h"
#include "VecTT.h"

namespace sdot {

VecTT::VecTT( KernelSlot *ks, BI rese, BI size ) : ks( ks ) {
    if ( rese )
        _data = ks->allocate_TT( rese );
    _rese = rese;
    _size = size;
}

VecTT::VecTT( const VecTT &that ) : VecTT( that.ks, that.size(), that.size() ) {
    ks->assign_TT( _data, 0, that._data, 0, _size );
}

VecTT::VecTT( VecTT &&that ) : ks( that.ks ) {
    _data = std::exchange( that._data, nullptr );
    _rese = std::exchange( that._rese, 0 );
    _size = std::exchange( that._size, 0 );
}

VecTT::~VecTT() {
    free();
}

VecTT &VecTT::operator=( const VecTT &that ) {
    resize( that.size() );
    ks->assign_TT( _data, 0, that._data, 0, _size );
    return *this;
}

VecTT &VecTT::operator=( VecTT &&that ) {
    free();

    _data = std::exchange( that._data, nullptr );
    _rese = std::exchange( that._rese, 0 );
    _size = std::exchange( that._size, 0 );
    ks = that.ks;

    return *this;
}

void VecTT::write_to_stream( std::ostream &os ) const {
    ks->display_TT( os, _data, 0, _size );
}

void VecTT::display( std::ostream &os, BI off, BI len ) const {
    ks->display_TT( os, data(), off, len );
}

VecTT::BI VecTT::size() const {
    return _size;
}

const void *VecTT::data() const {
    return _data;
}

void *VecTT::data() {
    return _data;
}

void VecTT::reserve( BI new_size, bool copy_if_resize ) {
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
    _data = ks->allocate_TT( _rese );
    if ( copy_if_resize )
        ks->assign_TT( _data, 0, old_data, 0, _size );
    if ( old_rese )
        ks->free_TT( old_data );
}

void VecTT::resize( BI new_size, bool copy_if_resize ) {
    reserve( new_size, copy_if_resize );
    _size = new_size;
}

void VecTT::free() {
    if ( _rese ) {
        ks->free_TT( _data );
        _rese = 0;
        _size = 0;
    }
}

} // namespace sdot

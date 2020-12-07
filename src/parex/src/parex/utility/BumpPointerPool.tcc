#include "BumpPointerPool.h"
#include <algorithm>
#include <cstdlib>
#include <memory>

namespace parex {

inline
BumpPointerPool::BumpPointerPool() {
    current_ptr.cp = nullptr;
    ending_ptr     = nullptr;
    last_frame     = nullptr;
    last_item      = nullptr;
}

inline
BumpPointerPool::~BumpPointerPool() {
    for( Item *f = last_item, *o; ( o = f ) ; ) {
        f = f->prev;
        o->~Item();
    }

    for( Frame *f = last_frame, *o; ( o = f ) ; ) {
        f = f->prev_frame;
        std::free( o );
    }
}

inline
char *BumpPointerPool::allocate( std::size_t size, std::size_t alig ) {
    using std::malloc;
    using std::max;

    // get aligned ptr
    current_ptr.vp = ( current_ptr.vp + alig - 1 ) & ~( alig - 1 );
    char *res = current_ptr.cp;

    // room
    current_ptr.cp += size;
    if ( current_ptr.cp > ending_ptr ) {
        std::size_t frame_size = max( 4096ul - 32ul, sizeof( Frame * ) + alig - 1 + size );
        Frame* new_frame = new ( malloc( frame_size ) ) Frame;
        new_frame->prev_frame = last_frame;
        last_frame = new_frame;

        ending_ptr = reinterpret_cast<char *>( new_frame ) + frame_size;
        current_ptr.cp = new_frame->content;

        current_ptr.vp = ( current_ptr.vp + alig - 1 ) & ~( alig - 1 );
        res = current_ptr.cp;

        current_ptr.cp += size;
    }

    return res;
}

template <class T,class... Args>
T* BumpPointerPool::create( Args &&...args ) {
    if ( std::is_trivially_destructible<T>::value )
        return new ( allocate( sizeof( T ), alignof( T ) ) ) T( std::forward<Args>( args )... );

    Inst<T> *item = new ( allocate( sizeof( Inst<T> ), alignof( Inst<T> ) ) ) Inst<T>( std::forward<Args>( args )... );
    item->prev = last_item;
    last_item = item;
    return &item->object;
}

} // namespace parex

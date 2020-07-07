#include "IntrusiveList.h"

template<class T>
IntrusiveList<T>::IntrusiveList( T *a, T *b ) {
    b->next = nullptr;
    a->next = b;
    data = a;
}

template<class T>
IntrusiveList<T>::IntrusiveList( T *a ) {
    a->next = nullptr;
    data = a;
}

template<class T>
IntrusiveList<T>::IntrusiveList() : data( nullptr ) {
}

template<class T>
void IntrusiveList<T>::push_front( T *item ) {
    item->next = data;
    data = item;
}

template<class T>
void IntrusiveList<T>::remove_if( const std::function<bool(T &)> &f ) {
    while ( data ) {
        T *next = data->next;
        if ( ! f( *data ) )
            break;
        data = next;
    }

    if ( T *curr = data ) {
        while ( curr->next ) {
            T *next = curr->next->next;
            if ( f( *curr->next ) )
                curr->next = next;
            else
                curr = curr->next;
        }
    }
}

template<class T>
void IntrusiveList<T>::move_to_if( IntrusiveList<T> &that, const std::function<bool( T & )> &cond ) {
    remove_if( [&]( T &val ) {
        if ( cond( val ) ) {
            that.push_front( &val );
            return true;
        }
        return false;
    } );
}

template<class T>
void IntrusiveList<T>::clear() {
    data = nullptr;
}

template<class T>
std::size_t IntrusiveList<T>::size() const {
    std::size_t res = 0;
    for( const T *c = data; c; c = c->next )
        ++res;
    return res;
}

template<class T>
T &IntrusiveList<T>::Iterator::operator*() const {
    return *ptr;
}

template<class T>
T *IntrusiveList<T>::Iterator::operator->() const {
    return ptr;
}

template<class T>
void IntrusiveList<T>::Iterator::operator++() {
    ptr = ptr->next;
}

template<class T>
bool IntrusiveList<T>::Iterator::operator!=( const Iterator &that) const {
    return ptr != that.ptr;
}

#include "IntrusiveList.h"

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
    while ( data && f( *data ) )
        data = data->next;

    if ( T *curr = data ) {
        while ( curr->next ) {
            if ( f( *curr->next ) )
                curr->next = curr->next->next;
            else
                curr = curr->next;
        }
    }
}

template<class T>
void IntrusiveList<T>::clear() {
    data = nullptr;
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

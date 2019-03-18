#include "PoolWithActiveAndInactiveItems.h"

template<class T>
PoolWithActiveAndInactiveItems<T>::PoolWithActiveAndInactiveItems( PoolWithActiveAndInactiveItems &&that ) : last_active( std::move( that.last_active ) ), pool( std::move( that.pool ) ) {
    that.last_active = nullptr;
}

template<class T>
PoolWithActiveAndInactiveItems<T>::PoolWithActiveAndInactiveItems() {
    last_active = nullptr;
}

template<class T>
void PoolWithActiveAndInactiveItems<T>::operator=( PoolWithActiveAndInactiveItems &&that ) {
    last_active = that.last_active;
    that.last_active = nullptr;
    pool = std::move( that.pool );
}

template<class T>
void PoolWithActiveAndInactiveItems<T>::free( T *item ) {
    // remove from the active list
    if ( item->next_in_pool )
        item->next_in_pool->prev_in_pool = item->prev_in_pool;
    else
        last_active = item->prev_in_pool;
    if ( item->prev_in_pool )
        item->prev_in_pool->next_in_pool = item->next_in_pool;

    // append in the inactive list
    pool.free( item );
}

template<class T>
T *PoolWithActiveAndInactiveItems<T>::new_item() {
    T *res = pool.new_item();

    // append in the active list
    res->prev_in_pool = last_active;
    res->next_in_pool = nullptr;
    if ( last_active )
        last_active->next_in_pool = res;
    last_active = res;

    return res;
}

template<class T>
bool PoolWithActiveAndInactiveItems<T>::empty() const {
    return last_active == nullptr;
}

template<class T>
void PoolWithActiveAndInactiveItems<T>::clear() {
    while ( last_active ) {
        T *prev = last_active->prev_in_pool;
        pool.free( last_active );
        last_active = prev;
    }
}

template<class T>
std::size_t PoolWithActiveAndInactiveItems<T>::size() const {
    return pool.size();
}

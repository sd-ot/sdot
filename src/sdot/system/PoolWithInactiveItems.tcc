#include "PoolWithInactiveItems.h"

template<class T>
PoolWithInactiveItems<T>::PoolWithInactiveItems() {
    last_inactive = nullptr;
}

template<class T>
void PoolWithInactiveItems<T>::free( T *item ) {
    item->prev_in_pool = last_inactive;
    last_inactive = item;
}

template<class T>
T *PoolWithInactiveItems<T>::new_item() {
    // we have at least one inactive item ?
    if ( last_inactive ) {
        // remove from the inactive list
        T *res = last_inactive;
        last_inactive = res->prev_in_pool;
        return res;
    }

    // create a new item
    content.emplace_back();
    T *res = &content.back();
    return res;
}

template<class T>
void PoolWithInactiveItems<T>::clear() {
    last_inactive = nullptr;
    for( T &item : content ) {
        item.prev_in_pool = last_inactive;
        last_inactive = &item;
    }
}

template<class T>
std::size_t PoolWithInactiveItems<T>::size() const {
    std::size_t res = content.size();
    for( T *p = last_inactive; p; p = p->prev_in_pool )
        --res;
    return res;
}

#include "PoolWithInactiveItems.h"

template<class T>
PoolWithInactiveItems<T>::PoolWithInactiveItems() {
    last_inactive = nullptr;
    last_active   = nullptr;
}

template<class T>
void PoolWithInactiveItems<T>::free( T *item ) {
    last_active = item->prev_in_pool;
    item->prev_in_pool = last_inactive;
    last_inactive = item;
}

template<class T>
T *PoolWithInactiveItems<T>::get_item() {
    if ( last_inactive ) {
        T *res = last_inactive;
        last_inactive = res->prev_in_pool;
        res->prev_in_pool = last_active;
        last_active = res;
        return res;
    }
    content.emplace_back();
    T *res = &content.back();
    res->prev_in_pool = last_active;
    last_active = res;
    return res;
}

#include "PoolWithInactiveItems.h"

template<class T>
PoolWithInactiveItems<T>::PoolWithInactiveItems() {
    last_inactive = nullptr;
    last_active   = nullptr;
}

template<class T>
void PoolWithInactiveItems<T>::free( T *item ) {
    // remove from active list
    if ( item->next_in_pool )
        item->next_in_pool->prev_in_pool = item->prev_in_pool;
    else
        last_active = item->prev_in_pool;
    if ( item->prev_in_pool )
        item->prev_in_pool->next_in_pool = item->next_in_pool;

    // append in inactive list
    item->prev_in_pool = last_inactive;
    last_inactive = item;
}

template<class T>
T *PoolWithInactiveItems<T>::get_item() {
    // we have at least one inactive item ?
    if ( last_inactive ) {
        // remove from inactive list
        T *res = last_inactive;
        last_inactive = res->prev_in_pool;

        // append in active list
        res->prev_in_pool = last_active;
        res->next_in_pool = nullptr;
        if ( last_active )
            last_active->next_in_pool = res;
        last_active = res;
        return res;
    }

    // create a new item
    content.emplace_back();
    T *res = &content.back();

    // append it in active list
    res->prev_in_pool = last_active;
    res->next_in_pool = nullptr;
    if ( last_active )
        last_active->next_in_pool = res;
    last_active = res;
    return res;
}

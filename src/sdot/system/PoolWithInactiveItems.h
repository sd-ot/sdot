#pragma once

#include <deque>

/**
  T is expected to contain
  - a `T *prev_in_pool` attribute (to have lists of active and inactive items).
  - a `T *next_in_pool` attribute (to have lists of active items).
*/
template<class T>
class PoolWithInactiveItems {
public:
    /**/          PoolWithInactiveItems();

    T            *get_item             ();
    void          free                 ( T *item );

private:
    T            *last_inactive;
    T            *last_active;
    std::deque<T> content;
};

#include "PoolWithInactiveItems.tcc"

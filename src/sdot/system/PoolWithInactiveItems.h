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

    T            *new_item             ();
    void          clear                ();
    void          free                 ( T *item );

    std::size_t   size                 () const;

private:
    T            *last_inactive;
    std::deque<T> content;
};

#include "PoolWithInactiveItems.tcc"


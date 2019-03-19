#pragma once

#include "PoolWithInactiveItems.h"

/**
  A PoolWithInactiveItems that keeps track of active items (allowing to have and iterator, ...)
*/
template<class T>
class PoolWithActiveAndInactiveItems {
public:
    struct                   Iterator {
        /**/                 Iterator  ( T *current ) : current( current ) {}
        bool                 operator!=( const Iterator &that ) const { return current != that.current; }
        void                 operator++() { current = current->prev_in_pool; }
        T                   *operator->() const { return current; }
        T                   &operator* () const { return *current; }

        T                   *current;
    };

    /**/                     PoolWithActiveAndInactiveItems( const PoolWithActiveAndInactiveItems &that ) = delete;
    /**/                     PoolWithActiveAndInactiveItems( PoolWithActiveAndInactiveItems &&that );
    /**/                     PoolWithActiveAndInactiveItems();

    void                     operator=                     ( const PoolWithActiveAndInactiveItems &that ) = delete;
    void                     operator=                     ( PoolWithActiveAndInactiveItems &&that );

    template                 <class OS>
    void                     write_to_stream               ( OS &os, const char *sep = " " ) const { int cpt = 0; for( const T &val : *this ) os << ( cpt++ ? sep : "" ) << val; }

    // modifications
    T                       *new_item                      ();
    void                     clear                         ();
    void                     free                          ( T *item );

    // information
    bool                     empty                         () const;
    std::size_t              size                          () const;

    Iterator                 begin                         () const { return last_active; }
    Iterator                 end                           () const { return nullptr; }

private:
    T                       *last_active;
    PoolWithInactiveItems<T> pool;
};

#include "PoolWithActiveAndInactiveItems.tcc"


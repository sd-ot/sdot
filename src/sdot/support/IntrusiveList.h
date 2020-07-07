#ifndef INTRUSIVELIST_H
#define INTRUSIVELIST_H

#include <functional>

/**
*/
template<class T>
struct IntrusiveList {
    struct      Iterator     { T *ptr; T &operator*() const; T *operator->() const; void operator++(); bool operator!=( const Iterator &that ) const; };

    /**/        IntrusiveList( T *a, T *b );
    /**/        IntrusiveList( T *a );
    /**/        IntrusiveList();

    void        push_front   ( T *item );
    void        move_to_if   ( IntrusiveList<T> &that, const std::function<bool( T & )> &cond );
    void        remove_if    ( const std::function<bool( T & )> &cond ); /// true to remove
    void        clear        ();

    std::size_t size         () const;

    Iterator    begin        () const { return { data }; }
    Iterator    end          () const { return { nullptr }; }

    const T&    front        () const { return *data; }
    T&          front        () { return *data; }

    const T&    first        () const { return *data; }
    T&          first        () { return *data; }

    const T&    second       () const { return *data->next; }
    T&          second       () { return *data->next; }

    bool        empty        () const { return ! data; }

    T*          data;
};

#include "IntrusiveList.tcc"

#endif // INTRUSIVELIST_H

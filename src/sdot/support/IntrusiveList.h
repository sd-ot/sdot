#ifndef INTRUSIVELIST_H
#define INTRUSIVELIST_H

#include <functional>

/**
*/
template<class T>
struct IntrusiveList {
    struct   Iterator     { T *ptr; T &operator*() const; T *operator->() const; void operator++(); bool operator!=( const Iterator &that ) const; };

    /**/     IntrusiveList();

    void     push_front   ( T *item );
    void     remove_if    ( const std::function<bool( T & )> &f ); /// true to remove
    void     clear        ();

    Iterator begin        () const { return { data }; }
    Iterator end          () const { return { nullptr }; }

    bool     empty        () const { return ! data; }

    T*       data;
};

#include "IntrusiveList.tcc"

#endif // INTRUSIVELIST_H

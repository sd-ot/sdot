#ifndef INTRUSIVELIST_H
#define INTRUSIVELIST_H

/**
*/
template<class T>
struct IntrusiveList {
    struct   Iterator     { T *ptr; T &operator*() const { return *ptr; } T *operator->() const { return ptr; } void operator++() { ptr = ptr->next; } bool operator!=( const Iterator &that ) const { return ptr != that.ptr; } };

    /**/     IntrusiveList() : data( nullptr ) {}

    void     push_front   ( T *item ) { item->next = data; data = item; }

    Iterator begin        () const { return { data }; }
    Iterator end          () const { return { nullptr }; }

    bool     empty        () const { return ! data; }

    T*       data;
};

#endif // INTRUSIVELIST_H

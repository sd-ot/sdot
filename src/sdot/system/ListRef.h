#pragma once

#include <ostream>

/**
  GetNext must return a reference on the pointer (allowing modifications)
*/
template<class T,class GetNext>
class ListRef {
public:
    struct Iterator {
        /**/      Iterator       ( T *current ) : current( current ) {}
        bool      operator!=     ( const Iterator &that ) const { return current != that.current; }
        Iterator &operator++     () { current = get_next( current ); return *this; }
        T        &operator*      () const { return *current; }
        T        *operator->     () const { return current; }

        GetNext   get_next;      ///<
        T        *current;       ///<
    };

    /**/          ListRef        () : first_item( nullptr ), last_item( nullptr ) {}

    void          write_to_stream( std::ostream &os, const char *sep = " " ) const { int cpt = 0; for( const T &val : *this ) os << ( cpt++ ? sep : "" ) << val; }
    Iterator      begin          () const { return first_item; }
    Iterator      end            () const { return nullptr; }

    void          insert_between ( T *prev, T *next, T *item ) { get_next( prev ) = item; get_next( item ) = next; }
    void          append         ( T *item ) { if ( last_item ) get_next( last_item ) = item; else first_item = item; last_item = item; get_next( item ) = nullptr; }
    void          clear          () { first_item = nullptr; last_item = nullptr; }

private:
    T            *first_item;    ///<
    T            *last_item;     ///<
    GetNext       get_next;
};

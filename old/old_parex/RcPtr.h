#ifndef PAREX_RcPtr_H
#define PAREX_RcPtr_H

#include "generic_ostream_output.h"
#include "Delete.h"
#include "Free.h"

/**
 *
 */
template<class T,class DeleteMethod=Delete>
struct RcPtr {
    /**/                     RcPtr          ();
    /**/                     RcPtr          ( T *obj );
    /**/                     RcPtr          ( RcPtr &&obj );
    template<class U>        RcPtr          ( RcPtr<U> &&obj );
    /**/                     RcPtr          ( const RcPtr &obj );
    template<class U>        RcPtr          ( const RcPtr<U> &obj );

    /**/                    ~RcPtr          ();

    void                     write_to_stream( std::ostream &os ) const;
    T                       *operator->     () const;
    T                       &operator*      () const;
    explicit operator        bool           () const;
    T                       *ptr            () const;

    void                     clear          ();

    RcPtr                   &operator=      ( T *ptr );
    template<class U> RcPtr &operator=      ( U *ptr );
    RcPtr                   &operator=      ( RcPtr &&obj );
    template<class U> RcPtr &operator=      ( RcPtr<U> &&obj );
    RcPtr                   &operator=      ( const RcPtr &obj );
    template<class U> RcPtr &operator=      ( const RcPtr<U> &obj );


    bool                     operator==     ( const T            *p ) const;
    bool                     operator==     ( const RcPtr<T>     &p ) const;
    // bool                  operator==     ( const ConstPtr<T> &p ) const { return data == p.data; }

    bool                     operator!=     ( const T            *p ) const;
    bool                     operator!=     ( const RcPtr<T>     &p ) const;
    // bool                  operator!=     ( const ConstPtr<T> &p ) const { return data != p.data; }

    bool                     operator<      ( const T            *p ) const;
    bool                     operator<      ( const RcPtr<T>     &p ) const;
    // bool                  operator<      ( const ConstPtr<T> &p ) const { return data <  p.data; }

    bool                     operator<=     ( const T            *p ) const;
    bool                     operator<=     ( const RcPtr<T>     &p ) const;
    // bool                  operator<=     ( const ConstPtr<T> &p ) const { return data <= p.data; }

    bool                     operator>      ( const T            *p ) const;
    bool                     operator>      ( const RcPtr<T>     &p ) const;
    // bool                  operator>      ( const ConstPtr<T> &p ) const { return data >  p.data; }

    bool                     operator>=     ( const T            *p ) const;
    bool                     operator>=     ( const RcPtr<T>     &p ) const;
    // bool                  operator>=     ( const ConstPtr<T> &p ) const { return data >= p.data; }

    void                     inc_ref( T *data );
    void                     dec_ref( T *data );

    DeleteMethod             delete_method;
    T                       *data;
};

template<class T>
bool operator==( const T *p, const RcPtr<T> &q );

#include "RcPtr.tcc"

#endif // PAREX_RcPtr_H

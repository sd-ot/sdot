#ifndef PAREX_Rc_H
#define PAREX_Rc_H

#include "generic_ostream_output.h"
#include "Delete.h"
#include "Free.h"

/**
 *
 */
template<class T,class DeleteMethod=Delete>
struct Rc {
    template<class U>     Rc             ( const Rc<U> &obj );
    /**/                  Rc             ( const Rc &obj );
    template<class U>     Rc             ( Rc<U> &&obj );
    /**/                  Rc             ( Rc &&obj );
    /**/                  Rc             ( T *obj );
    /**/                  Rc             ();

    /**/                 ~Rc             ();

    void                  write_to_stream( std::ostream &os ) const;
    T*                    operator->     () const;
    T&                    operator*      () const;
    explicit operator     bool           () const;
    T*                    ptr            () const;

    void                  clear          ();

    template<class U> Rc& operator=      ( const Rc<U> &obj );
    Rc&                   operator=      ( const Rc &obj );
    template<class U> Rc& operator=      ( Rc<U> &&obj );
    Rc&                   operator=      ( Rc &&obj );
    Rc &                  operator=      ( T *ptr );
    template<class U> Rc& operator=      ( U *ptr );

    bool                  operator==     ( const T     *p ) const;
    bool                  operator==     ( const Rc<T> &p ) const;
    bool                  operator!=     ( const T     *p ) const;
    bool                  operator!=     ( const Rc<T> &p ) const;
    bool                  operator<      ( const T     *p ) const;
    bool                  operator<      ( const Rc<T> &p ) const;
    bool                  operator<=     ( const T     *p ) const;
    bool                  operator<=     ( const Rc<T> &p ) const;
    bool                  operator>      ( const T     *p ) const;
    bool                  operator>      ( const Rc<T> &p ) const;
    bool                  operator>=     ( const T     *p ) const;
    bool                  operator>=     ( const Rc<T> &p ) const;

    void                  inc_ref        ( T *data );
    void                  dec_ref        ( T *data );

    DeleteMethod          delete_method; ///<
    T*                    data;          ///<
};

template<class T>
bool operator==( const T *p, const Rc<T> &q );

#include "Rc.tcc"

#endif // PAREX_Rc_H

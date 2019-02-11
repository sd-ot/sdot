#pragma once

#include "Stream.h"

/**
 *
 */
template<class T, bool use_malloc=false>
struct RcPtr {
    RcPtr() : data( 0 ) {}
    RcPtr( T *obj ) : data( obj ) { if ( data ) ++data->cpt_use; }
    RcPtr( RcPtr &&obj ) : data( obj.data ) { obj.data = 0; }
    RcPtr( const RcPtr &obj ) : data( obj.data ) { if ( data ) ++data->cpt_use; }

    template<class U>
    RcPtr( const RcPtr<U> &obj ) : data( obj.data ) { if ( data ) ++data->cpt_use; }

    template<class U>
    RcPtr( RcPtr<U> &&obj ) : data( obj.data ) { obj.data = 0; }

    ~RcPtr() {
        if ( data && --data->cpt_use == 0 )
            _free( data );
    }

    void _free( T *data ) {
        if ( use_malloc ) {
            data->~T();
            free( data );
        } else
            delete data;
    }

    RcPtr &operator=( T *obj ) {
        if ( obj )
            ++obj->cpt_use;
        if ( data && --data->cpt_use == 0 )
            _free( data );
        data = obj;
        return *this;
    }

    template<class U>
    RcPtr &operator=( U *obj ) {
        if ( obj )
            ++obj->cpt_use;
        if ( data && --data->cpt_use == 0 )
            _free( data );
        data = obj;
        return *this;
    }

    RcPtr &operator=( const RcPtr &obj ) {
        if ( obj.data )
            ++obj.data->cpt_use;
        if ( data && --data->cpt_use == 0 )
            _free( data );
        data = obj.data;
        return *this;
    }

    template<class U>
    RcPtr &operator=( const RcPtr<U> &obj ) {
        if ( obj.data )
            ++obj.data->cpt_use;
        if ( data && --data->cpt_use == 0 )
            _free( data );
        data = obj.data;
        return *this;
    }

    RcPtr &operator=( RcPtr &&obj ) {
        if ( data && data != obj.data && --data->cpt_use == 0 )
            _free( data );
        data = obj.data;
        obj.data = 0;
        return *this;
    }

    template<class U>
    RcPtr &operator=( RcPtr<U> &&obj ) {
        if ( data && data != obj.data && --data->cpt_use == 0 )
            _free( data );
        data = obj.data;
        obj.data = 0;
        return *this;
    }


    explicit operator bool() const { return data; }

    void clear() { if ( data ) { if ( --data->cpt_use == 0 ) _free( data ); data = nullptr; } }

    bool operator==( const T            *p ) const { return data == p;      }
    bool operator==( const RcPtr<T>     &p ) const { return data == p.data; }
    // bool operator==( const ConstPtr<T> &p ) const { return data == p.data; }

    bool operator!=( const T            *p ) const { return data != p;      }
    bool operator!=( const RcPtr<T>     &p ) const { return data != p.data; }
    // bool operator!=( const ConstPtr<T> &p ) const { return data != p.data; }

    bool operator< ( const T            *p ) const { return data <  p;      }
    bool operator< ( const RcPtr<T>     &p ) const { return data <  p.data; }
    // bool operator< ( const ConstPtr<T> &p ) const { return data <  p.data; }

    bool operator<=( const T            *p ) const { return data <= p;      }
    bool operator<=( const RcPtr<T>     &p ) const { return data <= p.data; }
    // bool operator<=( const ConstPtr<T> &p ) const { return data <= p.data; }

    bool operator> ( const T            *p ) const { return data >  p;      }
    bool operator> ( const RcPtr<T>     &p ) const { return data >  p.data; }
    // bool operator> ( const ConstPtr<T> &p ) const { return data >  p.data; }

    bool operator>=( const T            *p ) const { return data >= p;      }
    bool operator>=( const RcPtr<T>     &p ) const { return data >= p.data; }
    // bool operator>=( const ConstPtr<T> &p ) const { return data >= p.data; }

    T *ptr() const { return data; }

    T *operator->() const { return data; }

    T &operator*() const { return *data; }

    void write_to_stream( std::ostream &os ) const {
        if ( data )
            os << *data;
        else
            os << "NULL";
    }

    T *data;
};

/**
 *
 */
struct RcObj {
    RcObj() : cpt_use( 0 ) {}
    mutable size_t cpt_use;
};

template<class T>
inline const T *inc_ref( const T *p ) {
    ++p->cpt_use;
    return p;
}

template<class T>
inline T *inc_ref( T *p ) {
    ++p->cpt_use;
    return p;
}

//template<class T>
//inline void dec_ref( const T *ptr ) {
//    if ( --ptr->cpt_use == 0 )
//        ptr->_free();
//}

template<class T>
bool operator==( const T *p, const RcPtr<T> &q ) { return p == q.data; }

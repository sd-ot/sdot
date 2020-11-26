#include "RcPtr.h"

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod>::RcPtr() : data( 0 ) {
}

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod>::RcPtr( T *obj ) : data( obj ) {
    inc_ref( data );
}

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod>::RcPtr( RcPtr &&obj ) : data( std::exchange( obj.data, nullptr ) ) {
}

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod>::RcPtr( const RcPtr &obj ) : data( obj.data ) {
    inc_ref( data );
}

template<class T,class DeleteMethod> template<class U>
RcPtr<T,DeleteMethod>::RcPtr( const RcPtr<U> &obj ) : data( obj.data ) {
    inc_ref( data );
}

template<class T,class DeleteMethod> template<class U>
RcPtr<T,DeleteMethod>::RcPtr( RcPtr<U> &&obj ) : data( std::exchange( obj.data, nullptr ) ) {
}

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod>::~RcPtr() {
    dec_ref( data );
}

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod> &RcPtr<T,DeleteMethod>::operator=( T *atad ) {
    inc_ref( atad );
    dec_ref( data );
    data = atad;
    return *this;
}

template<class T,class DeleteMethod> template<class U>
RcPtr<T,DeleteMethod> &RcPtr<T,DeleteMethod>::operator=( U *atad ) {
    inc_ref( atad );
    dec_ref( data );
    data = atad;
    return *this;
}

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod> &RcPtr<T,DeleteMethod>::operator=( const RcPtr &obj ) {
    inc_ref( obj.data );
    dec_ref( data );
    data = obj.data;
    return *this;
}

template<class T,class DeleteMethod> template<class U>
RcPtr<T,DeleteMethod> &RcPtr<T,DeleteMethod>::operator=( const RcPtr<U> &obj ) {
    inc_ref( obj.data );
    dec_ref( data );
    data = obj.data;
    return *this;
}

template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod> &RcPtr<T,DeleteMethod>::operator=( RcPtr &&obj ) {
    dec_ref( data );
    data = std::exchange( obj.data, nullptr );
    return *this;
}

template<class T,class DeleteMethod> template<class U>
RcPtr<T,DeleteMethod> &RcPtr<T,DeleteMethod>::operator=( RcPtr<U> &&obj ) {
    dec_ref( data );
    data = std::exchange( obj.data, nullptr );
    return *this;
}


template<class T,class DeleteMethod>
RcPtr<T,DeleteMethod>::operator bool() const {
    return data;
}

template<class T,class DeleteMethod>
void RcPtr<T,DeleteMethod>::clear() {
    dec_ref( std::exchange( data, nullptr ) );
}

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator==( const T            *p ) const { return data == p;      }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator==( const RcPtr<T>     &p ) const { return data == p.data; }
// bool operator==( const ConstPtr<T> &p ) const { return data == p.data; }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator!=( const T            *p ) const { return data != p;      }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator!=( const RcPtr<T>     &p ) const { return data != p.data; }
// bool operator!=( const ConstPtr<T> &p ) const { return data != p.data; }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator< ( const T            *p ) const { return data <  p;      }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator< ( const RcPtr<T>     &p ) const { return data <  p.data; }
// bool operator< ( const ConstPtr<T> &p ) const { return data <  p.data; }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator<=( const T            *p ) const { return data <= p;      }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator<=( const RcPtr<T>     &p ) const { return data <= p.data; }
// bool operator<=( const ConstPtr<T> &p ) const { return data <= p.data; }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator> ( const T            *p ) const { return data >  p;      }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator> ( const RcPtr<T>     &p ) const { return data >  p.data; }
// bool operator> ( const ConstPtr<T> &p ) const { return data >  p.data; }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator>=( const T            *p ) const { return data >= p;      }

template<class T,class DeleteMethod>
bool RcPtr<T,DeleteMethod>::operator>=( const RcPtr<T>     &p ) const { return data >= p.data; }
// bool operator>=( const ConstPtr<T> &p ) const { return data >= p.data; }

template<class T,class DeleteMethod>
T *RcPtr<T,DeleteMethod>::ptr() const { return data; }

template<class T,class DeleteMethod>
T *RcPtr<T,DeleteMethod>::operator->() const { return data; }

template<class T,class DeleteMethod>
T &RcPtr<T,DeleteMethod>::operator*() const { return *data; }

template<class T,class DeleteMethod>
void RcPtr<T,DeleteMethod>::write_to_stream( std::ostream &os ) const { if ( data ) os << *data; else os << "NULL"; }

template<class T,class DeleteMethod>
void RcPtr<T,DeleteMethod>::inc_ref( T *data ) { if ( data ) data->ref_count.increment(); }

template<class T,class DeleteMethod>
void RcPtr<T,DeleteMethod>::dec_ref( T *data ) { if ( data && data->ref_count.decrement() ) delete_method( data ); }

template<class T>
bool operator==( const T *p, const RcPtr<T> &q ) { return p == q.data; }

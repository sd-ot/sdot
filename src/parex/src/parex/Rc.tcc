#include "Rc.h"

template<class T,class DeleteMethod>
Rc<T,DeleteMethod>::Rc() : data( 0 ) {
}

template<class T,class DeleteMethod>
Rc<T,DeleteMethod>::Rc( T *obj ) : data( obj ) {
    inc_ref( data );
}

template<class T,class DeleteMethod>
Rc<T,DeleteMethod>::Rc( Rc &&obj ) : data( std::exchange( obj.data, nullptr ) ) {
}

template<class T,class DeleteMethod>
Rc<T,DeleteMethod>::Rc( const Rc &obj ) : data( obj.data ) {
    inc_ref( data );
}

template<class T,class DeleteMethod> template<class U>
Rc<T,DeleteMethod>::Rc( const Rc<U> &obj ) : data( obj.data ) {
    inc_ref( data );
}

template<class T,class DeleteMethod> template<class U>
Rc<T,DeleteMethod>::Rc( Rc<U> &&obj ) : data( std::exchange( obj.data, nullptr ) ) {
}

template<class T,class DeleteMethod>
Rc<T,DeleteMethod>::~Rc() {
    dec_ref( data );
}

template<class T,class DeleteMethod>
Rc<T,DeleteMethod> &Rc<T,DeleteMethod>::operator=( T *atad ) {
    inc_ref( atad );
    dec_ref( data );
    data = atad;
    return *this;
}

template<class T,class DeleteMethod> template<class U>
Rc<T,DeleteMethod> &Rc<T,DeleteMethod>::operator=( U *atad ) {
    inc_ref( atad );
    dec_ref( data );
    data = atad;
    return *this;
}

template<class T,class DeleteMethod>
Rc<T,DeleteMethod> &Rc<T,DeleteMethod>::operator=( const Rc &obj ) {
    inc_ref( obj.data );
    dec_ref( data );
    data = obj.data;
    return *this;
}

template<class T,class DeleteMethod> template<class U>
Rc<T,DeleteMethod> &Rc<T,DeleteMethod>::operator=( const Rc<U> &obj ) {
    inc_ref( obj.data );
    dec_ref( data );
    data = obj.data;
    return *this;
}

template<class T,class DeleteMethod>
Rc<T,DeleteMethod> &Rc<T,DeleteMethod>::operator=( Rc &&obj ) {
    dec_ref( data );
    data = std::exchange( obj.data, nullptr );
    return *this;
}

template<class T,class DeleteMethod> template<class U>
Rc<T,DeleteMethod> &Rc<T,DeleteMethod>::operator=( Rc<U> &&obj ) {
    dec_ref( data );
    data = std::exchange( obj.data, nullptr );
    return *this;
}


template<class T,class DeleteMethod>
Rc<T,DeleteMethod>::operator bool() const {
    return data;
}

template<class T,class DeleteMethod>
void Rc<T,DeleteMethod>::clear() {
    dec_ref( std::exchange( data, nullptr ) );
}

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator==( const T *p ) const { return data == p; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator==( const Rc<T> &p ) const { return data == p.data; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator!=( const T *p ) const { return data != p; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator!=( const Rc<T> &p ) const { return data != p.data; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator< ( const T *p ) const { return data <  p; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator< ( const Rc<T> &p ) const { return data <  p.data; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator<=( const T *p ) const { return data <= p; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator<=( const Rc<T> &p ) const { return data <= p.data; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator> ( const T *p ) const { return data >  p; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator> ( const Rc<T> &p ) const { return data >  p.data; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator>=( const T *p ) const { return data >= p; }

template<class T,class DeleteMethod>
bool Rc<T,DeleteMethod>::operator>=( const Rc<T> &p ) const { return data >= p.data; }

template<class T,class DeleteMethod>
T *Rc<T,DeleteMethod>::ptr() const { return data; }

template<class T,class DeleteMethod>
T *Rc<T,DeleteMethod>::operator->() const { return data; }

template<class T,class DeleteMethod>
T &Rc<T,DeleteMethod>::operator*() const { return *data; }

template<class T,class DeleteMethod>
void Rc<T,DeleteMethod>::write_to_stream( std::ostream &os ) const { if ( data ) os << *data; else os << "NULL"; }

template<class T,class DeleteMethod>
void Rc<T,DeleteMethod>::inc_ref( T *data ) { if ( data ) data->ref_count.increment(); }

template<class T,class DeleteMethod>
void Rc<T,DeleteMethod>::dec_ref( T *data ) { if ( data && data->ref_count.decrement() ) delete_method( data ); }

template<class T>
bool operator==( const T *p, const Rc<T> &q ) { return p == q.data; }

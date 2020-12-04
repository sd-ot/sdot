#pragma once

#include <algorithm>
#include <vector>

/**
*/
template<class T>
class VecUnique : public std::vector<T> {
public:
    using std::vector<T>::vector;

    template<class U>
    bool contains( const U &value ) const {
        return std::find( this->begin(), this->end(), value ) != this->end();
    }

    template<class U>
    VecUnique &operator<<( U &&value ) {
        if ( ! contains( value ) )
            this->push_back( std::forward<U>( value ) );
        return *this;
    }

    template<class U>
    VecUnique &operator<<( const VecUnique<U> &values ) {
        for( const U &value : values )
            operator<<( value );
        return *this;
    }
};

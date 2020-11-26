#pragma once

#include <algorithm>
#include <vector>

/**
*/
template<class T>
class VecUnique : public std::vector<T> {
public:
    template<class U>
    VecUnique &operator<<( U &&value ) {
        if ( std::find( this->begin(), this->end(), value ) == this->end() )
            this->push_back( std::forward<U>( value ) );
        return *this;
    }
};

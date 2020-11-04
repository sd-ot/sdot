#include "Data.h"

namespace parex {

Data::Data( const Data &that ) {}

Data::Data( Data &&that ) {}

Data &Data::operator=( Data &&that ) {
    return *this;
}

Data &Data::operator=( const Data &that ) {
    return *this;
}

} // namespace parex


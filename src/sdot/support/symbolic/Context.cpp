#include "Context.h"

namespace Symbolic {

Context::Context() {
    date = 0;
}

Expr Context::named( std::string name ) {
    auto iter = nameds.find( name );
    if ( iter == nameds.end() )
        iter = nameds.insert( iter, { name, pool.create<Named>( this, name ) } );
    return iter->second;
}

Expr Context::number( Rational n ) {
    auto iter = numbers.find( n );
    if ( iter == numbers.end() )
        iter = numbers.insert( iter, { n, pool.create<Number>( this, n ) } );
    return iter->second;
}

}

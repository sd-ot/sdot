#include "Context.h"

namespace Symbolic {

Expr Context::named( std::string name ) {
    auto iter = nameds.find( name );
    if ( iter == nameds.end() )
        iter = nameds.insert( iter, { name, pool.create<Named>( this, name ) } );
    return iter->second;
}



}

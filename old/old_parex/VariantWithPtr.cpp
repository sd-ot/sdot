#include "VariantWithPtr.h"
#include "Destructors.h"

VariantWithPtr::VariantWithPtr( const Type &type, void *data, bool owned ) : Variant( type ), data( data ), owned( owned ) {
}

VariantWithPtr::~VariantWithPtr() {
    if ( owned )
        destructors.symbol_for<void(void *)>( type.name )( data );
}

#include "MapOfCompiledTaskLib.h"

CompiledTaskLib *MapOfCompiledTaskLib::lib( const Path &src_path, const std::vector<Type *> &children_types, int priority ) {
    Params key( src_path, children_types, priority );
    auto iter = map.find( key );
    if ( iter == map.end() )
        iter = map.emplace_hint( iter, key, CompiledTaskLib{ src_path, children_types, priority } );
    return &iter->second;
}

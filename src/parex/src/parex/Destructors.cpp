#include "TypeInfoFactory.h"
#include "Destructors.h"

Destructors destructors;

CompiledSymbolMap::Path Destructors::output_directory( const std::string & ) const {
    return ".obj/destroy";
}

void Destructors::make_srcs( SrcWriter &ff ) const {
    TypeInfo *type_info = type_info_factory( ff.parameters );

    type_info->get_includes( [&]( const std::string &include ) { ff << "#include " << include << "\n"; } );

    type_info->get_preliminaries( [&]( const std::string &preliminary ) { ff << preliminary << "\n"; } );

    ff << "extern \"C\" void " << ff.symbol_name << "( void *data ) { delete reinterpret_cast<" + ff.parameters + " *>( data ); }";
}

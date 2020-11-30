#include <parex/GeneratedSymbolSet.h>
#include <parex/P.h>

void test_lib() {
    GeneratedLibrarySet gls;
    DynamicLibrary *lib = gls.get_library( []( SrcSet &ss ) { ss.src() << "extern \"C\" int pouet() { return 17; };"; } );
    P( lib->symbol<int(void)>( "pouet" )() );
}

void test_sym() {
    GeneratedSymbolSet gis;
    auto *sym = gis.get_symbol<int(void)>( []( SrcSet &ss ) { ss.src() << "extern \"C\" int exported() { return 17; };"; } );
    P( sym() );
}

int main() {
    test_sym();
}

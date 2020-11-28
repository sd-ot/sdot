#include <parex/CompiledTask.h>
#include <parex/P.h>

int main() {
    GeneratedLibrarySet gls;
    DynamicLibrary *lib = gls.get_library( []( SrcSet &ss ) { ss.src() << "extern \"C\" int pouet() { return 17; };"; } );
    P( lib->symbol<int(void)>( "pouet" )() );
}

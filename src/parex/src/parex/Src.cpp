#include <algorithm>
#include "Src.h"

Src::Src( Path name, const CompilationEnvironment &compilation_environment ) : compilation_environment( compilation_environment ), name( name ) {
}

Src::Path Src::filename() const {
    if ( compilation_environment.cxx.contains( "nvcc" ) ) {
        Path c = name; c.replace_extension( ".cu" );
        return c;
    }
    return name;
}

void Src::write_to( std::ostream &os ) const {
    for( const auto &include : compilation_environment.includes )
        os << "#include " << include << "\n";

    if ( compilation_environment.preliminaries.size() ) {
        os << "\n";
        for( const auto &prelim : compilation_environment.preliminaries )
            os << prelim << "\n";
    }

    os << "\n";
    os << fout.str();
}

#include <algorithm>
#include "Src.h"

Src::Src( const CompilationEnvironment &compilation_environment ) : compilation_environment( compilation_environment ) {
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

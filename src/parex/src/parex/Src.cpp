#include <algorithm>
#include "Src.h"

Src::Src( VUPath include_directories, VUString cpp_flags, VUString includes, VUString prelims ) : include_directories( include_directories ), cpp_flags( cpp_flags ), includes( includes ), prelims( prelims ) {
}

void Src::write_to( std::ostream &os ) const {
    for( const auto &include : includes )
        os << "#include " << include << "\n";

    if ( prelims.size() ) {
        os << "\n";
        for( const auto &prelim : prelims )
            os << prelim << "\n";
    }

    os << "\n";
    os << fout.str();
}

#include <algorithm>
#include "Src.h"

Src::Src( VUS include_directories, VUS cpp_flags, VUS includes, VUS prelims ) : include_directories( include_directories ), cpp_flags( cpp_flags ), includes( includes ), prelims( prelims ) {
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
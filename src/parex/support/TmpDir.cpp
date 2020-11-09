#include "TmpDir.h"
#include <random>
#include "P.h"

namespace parex {

TmpDir::TmpDir( std::string basename ) {
    while ( true ) {
        std::string name = basename + std::to_string( std::rand() );
        p = std::filesystem::temp_directory_path() / name;

        std::error_code ec;
        if ( create_directory( p, ec ) )
            break;
    }
}

TmpDir::~TmpDir() {
    std::error_code ec;
    std::filesystem::remove_all( p, ec );
}

parex::TmpDir::operator std::string() const {
    return p.string();
}

} // namespace parex

#include "Type.h"
#include "Src.h"
#include "P.h"

Type::~Type() {
}

void Type::for_each_include_directory( const std::function<void (std::string)> &/*cb*/ ) const {
}

void Type::for_each_prelim( const std::function<void(std::string)> &/*cb*/ ) const {
}

void Type::for_each_include( const std::function<void(std::string)> &/*cb*/ ) const {
}

void Type::add_needs_in( Src &src ) const {
    src.include_directories << SDOT_DIR "/src/parex/src";

    for_each_include_directory( [&]( std::string p ) { src.include_directories << p; } );
    for_each_include( [&]( std::string p ) { src.includes << p; } );
    for_each_prelim( [&]( std::string p ) { src.prelims << p; } );
}

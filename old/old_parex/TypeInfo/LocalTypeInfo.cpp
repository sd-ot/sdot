#include "LocalTypeInfo.h"

LocalTypeInfo::LocalTypeInfo( const std::string &name, const std::vector<std::string> &includes, const std::string &preliminary ) : preliminary( preliminary ), includes( includes ), name( name ) {
}

void LocalTypeInfo::get_preliminaries( const std::function<void (const std::string &)> &f ) {
    if ( preliminary.size() )
        f( preliminary );
}

void LocalTypeInfo::get_includes( const std::function<void (const std::string &)> &f ) {
    for( const std::string &include : includes )
        f( include );
}

std::string LocalTypeInfo::cpp_name() {
    return name;
}

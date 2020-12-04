#include "DynamicLibrary.h"

namespace parex {

DynamicLibrary::DynamicLibrary( const Path &path ) : lib( path.string() ) {
}

} // namespace parex

#include "DynamicLibrary.h"

DynamicLibrary::DynamicLibrary( const Path &path ) : lib( path.string() ) {
}

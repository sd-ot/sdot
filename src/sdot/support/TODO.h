#pragma once

#include <iostream>
#include <assert.h>

void _TODO( const char *file, int line ) {
    std::cerr << file << ":" << line << ": TODO; ";
    assert( 0 );
}

#define TODO \
    _TODO( __FILE__, __LINE__ )

#pragma once

#include <iostream>
#include <assert.h>

#define TODO \
    do { std::cerr << __FILE__ << ":" << __LINE__ << ": TODO; "; assert( 0 ); } while ( 0 )

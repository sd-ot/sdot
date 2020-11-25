#pragma once

#include "ERROR.h"

#define TODO \
    do { std::cerr << __FILE__ << ":" << __LINE__ << ": TODO; "; assert( 0 ); } while ( 0 )

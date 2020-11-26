#pragma once

#include "va_string.h"
#include <iostream>
#include <assert.h>

#define ERROR( TXT, ... ) \
    do { std::cerr << va_string( TXT, ##__VA_ARGS__ ) << std::endl; assert( 0 ); } while ( 0 )

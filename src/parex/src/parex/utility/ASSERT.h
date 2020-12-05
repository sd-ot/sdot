#pragma once

#include "va_string.h"
#include <iostream>
#include <assert.h>

#define ASSERT( COND, TXT, ... ) \
    do { if ( ! ( COND ) ) { std::cerr << va_string( TXT, ##__VA_ARGS__ ) << std::endl; assert( 0 ); } } while ( 0 )

#define ASSERT_IF_DEBUG( COND )

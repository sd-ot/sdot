#pragma once

#include "TypeConfig.hpp"

void memcpy_bit( void *dst, size_t off_dst, const void *src, size_t off_src, size_t len, const void *msk = 0 ); ///< dst bits are sets only where msk bits are true. off_dst applies also on msk
void memcpy_bit( void *dst, const void *src, size_t len ); ///< same thing, without mask and offsets

void memset_bit( void *dst, size_t off_dst, bool val, size_t len );

int memcmp_bit( const void *dst, size_t off_dst, const void *src, size_t off_src, size_t len );
int memcmp_bit( const void *dst, size_t off, size_t len, bool val );

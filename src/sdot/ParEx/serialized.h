#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace parex {

std::string serialized_bin( std::string type, const void *data, std::size_t size );

inline std::string serialized( std::uint32_t val ) { return serialized_bin( "std::uint32_t", &val, sizeof( val ) ); }

inline std::string serialized( std::ostream* val ) { return serialized_bin( "std::ostream" , &val, sizeof( val ) ); }

} // namespace parex

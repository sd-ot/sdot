#pragma once

#include <cstdint>
#include <string>
#include "S.h"

inline std::string type_name( S<std::string  > ) { return "std::string"; }

inline std::string type_name( S<double       > ) { return "FP64"       ; }
inline std::string type_name( S<float        > ) { return "FP32"       ; }

inline std::string type_name( S<std::int8_t  > ) { return "SI8"        ; }
inline std::string type_name( S<std::int16_t > ) { return "SI16"       ; }
inline std::string type_name( S<std::int32_t > ) { return "SI32"       ; }
inline std::string type_name( S<std::int64_t > ) { return "SI64"       ; }

inline std::string type_name( S<std::uint8_t > ) { return "PI8"        ; }
inline std::string type_name( S<std::uint16_t> ) { return "PI16"       ; }
inline std::string type_name( S<std::uint32_t> ) { return "PI32"       ; }
inline std::string type_name( S<std::uint64_t> ) { return "PI64"       ; }

template<class T> std::string type_name( S<std::allocator<T>> ) { return "std::allocator<" + type_name( S<T>() ) + ">"; }

template<class T> std::string type_name() { return type_name( S<T>() ); }

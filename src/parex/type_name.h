#pragma once

#include <ostream>
#include <vector>
#include <string>

namespace parex {

inline std::string type_name( const std::ostream  * ) { return "ostream"; }
inline std::string type_name( const std::uint64_t * ) { return "PI64"   ; }
inline std::string type_name( const std::int32_t  * ) { return "SI32"   ; }
inline std::string type_name( const double        * ) { return "FP64"   ; }
inline std::string type_name( const float         * ) { return "FP32"   ; }
inline std::string type_name( const void          * ) { return "void"   ; }

} // namespace parex

#pragma once

#include "containers/Vec.h"
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

template<class T,class A>
inline std::string type_name( const Vec<T,A>      * ) { return "parex::Vec<" + type_name( reinterpret_cast<const T *>( 0ul ) ) + "," + A::name() + ">"   ; }

} // namespace parex

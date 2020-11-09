#pragma once

#include "containers/Tensor.h"
#include "containers/Vec.h"
#include <ostream>
#include <vector>
#include <string>
#include <tuple>


namespace parex {

inline std::string type_name( const std::ostream  * ) { return "ostream"; }
inline std::string type_name( const std::string   * ) { return "string" ; }

inline std::string type_name( const std::uint64_t * ) { return "PI64"   ; }
inline std::string type_name( const std::uint32_t * ) { return "PI32"   ; }
inline std::string type_name( const std::int64_t  * ) { return "SI64"   ; }
inline std::string type_name( const std::int32_t  * ) { return "SI32"   ; }
inline std::string type_name( const double        * ) { return "FP64"   ; }
inline std::string type_name( const float         * ) { return "FP32"   ; }
inline std::string type_name( const void          * ) { return "void"   ; }

template<class T,class A>
inline std::string type_name( const Vec<T,A>      * ) { return "parex::Vec<" + type_name( reinterpret_cast<const T *>( 0ul ) ) + ",parex::Arch::" + A::name() + ">"   ; }

template<class T,class A>
inline std::string type_name( const Tensor<T,A>   * ) { return "parex::Tensor<" + type_name( reinterpret_cast<const T *>( 0ul ) ) + ",parex::Arch::" + A::name() + ">"   ; }

} // namespace parex

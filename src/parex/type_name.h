#pragma once

#include <ostream>
#include <string>
#include <map>

namespace parex {

// declaration
template<class T> typename std::enable_if<!std::is_pointer<T>::value,std::string>::type type_name( const T * ) { return T::type_name(); }

inline                    std::string type_name( const std::ostream  * ) { return "std::ostream"; }
inline                    std::string type_name( const std::string   * ) { return "std::string" ; }

inline                    std::string type_name( const std::uint64_t * ) { return "PI64"   ; }
inline                    std::string type_name( const std::uint32_t * ) { return "PI32"   ; }
inline                    std::string type_name( const std::int64_t  * ) { return "SI64"   ; }
inline                    std::string type_name( const std::int32_t  * ) { return "SI32"   ; }
inline                    std::string type_name( const double        * ) { return "FP64"   ; }
inline                    std::string type_name( const float         * ) { return "FP32"   ; }

template<class T>         std::string type_name( const T * const     * ) { return type_name( reinterpret_cast<const T *>( 0ul ) ) + "*"; }

template<class T>         std::string type_name() { return type_name( reinterpret_cast<const T *>( 0ul ) ); }

template<class T,class A> std::string type_name( const std::map<T,A> * ) { return "std::map<" + type_name<T>() + "," + type_name<A>() + ">"; }


} // namespace parex

#pragma once

#include <ostream>
#include <string>
#include <array>
#include <map>

namespace parex {

// declarations
inline                          std::string type_name( const std::ostream    * ) { return "std::ostream"; }
inline                          std::string type_name( const std::string     * ) { return "std::string" ; }

inline                          std::string type_name( const std::uint64_t   * ) { return "PI64"   ; }
inline                          std::string type_name( const std::uint32_t   * ) { return "PI32"   ; }
inline                          std::string type_name( const std::int64_t    * ) { return "SI64"   ; }
inline                          std::string type_name( const std::int32_t    * ) { return "SI32"   ; }
inline                          std::string type_name( const double          * ) { return "FP64"   ; }
inline                          std::string type_name( const float           * ) { return "FP32"   ; }

template<class T,std::size_t d> std::string type_name( const std::array<T,d> * );
template<class T,class A>       std::string type_name( const std::map<T,A>   * );
template<class T>               std::string type_name( const T * const       * );
template<class T>               auto        type_name( const T               * ) -> typename std::enable_if<!std::is_pointer<T>::value,std::string>::type;
template<class T>               std::string type_name();

// definitions
template<class T,std::size_t d> std::string type_name( const std::array<T,d> * ) { return "std::array<" + type_name<T>() + "," + std::to_string( d ) + ">"; }
template<class T,class A>       std::string type_name( const std::map<T,A>   * ) { return "std::map<" + type_name<T>() + "," + type_name<A>() + ">"; }
template<class T>               std::string type_name( const T * const       * ) { return type_name( reinterpret_cast<const T *>( 0ul ) ) + "*"; }
template<class T>               auto        type_name( const T               * ) -> typename std::enable_if<!std::is_pointer<T>::value,std::string>::type { return T::type_name(); }
template<class T>               std::string type_name()                          { return type_name( reinterpret_cast<const T *>( 0ul ) ); }

} // namespace parex

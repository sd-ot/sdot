#pragma once

#include <string>
#include <array>
#include <map>

namespace parex {

template<class T>               struct TypeName                  { static std::string name() { return T::type_name() ; } };

template<>                      struct TypeName<std::string    > { static std::string name() { return "std::string"; } };

template<>                      struct TypeName<std::uint64_t  > { static std::string name() { return "PI64"       ; } };
template<>                      struct TypeName<std::uint32_t  > { static std::string name() { return "PI32"       ; } };
template<>                      struct TypeName<std::int64_t   > { static std::string name() { return "SI64"       ; } };
template<>                      struct TypeName<std::int32_t   > { static std::string name() { return "SI32"       ; } };
template<>                      struct TypeName<double         > { static std::string name() { return "FP64"       ; } };
template<>                      struct TypeName<float          > { static std::string name() { return "FP32"       ; } };

template<class T,std::size_t d> struct TypeName<std::array<T,d>> { static std::string name() { return "std::array<" + TypeName<T>::name() + "," + std::to_string( d ) + ">"; } };
template<class T,class A>       struct TypeName<std::map  <T,A>> { static std::string name() { return "std::map<"   + TypeName<T>::name() + "," + TypeName<A>::name() + ">"; } };

} // namespace parex

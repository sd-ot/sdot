#pragma once

#include <cstdint>

namespace sdot {

template<class T> struct TypeName;

template<> struct TypeName<double       > { static const char *name() { return "double"; } };
template<> struct TypeName<float        > { static const char *name() { return "float" ; } };

template<> struct TypeName<std::uint32_t> { static const char *name() { return "uint32"; } };
template<> struct TypeName<std::uint64_t> { static const char *name() { return "uint64"; } };

} // namespace sdot

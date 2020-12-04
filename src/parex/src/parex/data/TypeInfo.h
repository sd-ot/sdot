#pragma once

#include <cstdint>
#include <string>

namespace parex {

template<class T> struct TypeInfo;

template<> struct TypeInfo<std::string  > { static std::string name() { return "std::string"; } };

template<> struct TypeInfo<std::uint8_t > { static std::string name() { return "PI8"        ; } };
template<> struct TypeInfo<std::uint16_t> { static std::string name() { return "PI16"       ; } };
template<> struct TypeInfo<std::uint32_t> { static std::string name() { return "PI32"       ; } };
template<> struct TypeInfo<std::uint64_t> { static std::string name() { return "PI64"       ; } };

template<> struct TypeInfo<std::int8_t  > { static std::string name() { return "SI8"        ; } };
template<> struct TypeInfo<std::int16_t > { static std::string name() { return "SI16"       ; } };
template<> struct TypeInfo<std::int32_t > { static std::string name() { return "SI32"       ; } };
template<> struct TypeInfo<std::int64_t > { static std::string name() { return "SI64"       ; } };

template<> struct TypeInfo<float        > { static std::string name() { return "FP32"       ; } };
template<> struct TypeInfo<double       > { static std::string name() { return "FP64"       ; } };

} // namespace parex

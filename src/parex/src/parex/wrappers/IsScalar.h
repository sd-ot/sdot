#pragma once

#include <cstdint>

namespace parex {

template<class T> struct IsScalar { enum { value = false }; };

template<> struct IsScalar<std::uint8_t > { enum { value = true }; };
template<> struct IsScalar<std::uint16_t> { enum { value = true }; };
template<> struct IsScalar<std::uint32_t> { enum { value = true }; };
template<> struct IsScalar<std::uint64_t> { enum { value = true }; };

template<> struct IsScalar<std::int8_t  > { enum { value = true }; };
template<> struct IsScalar<std::int16_t > { enum { value = true }; };
template<> struct IsScalar<std::int32_t > { enum { value = true }; };
template<> struct IsScalar<std::int64_t > { enum { value = true }; };

template<> struct IsScalar<float        > { enum { value = true }; };
template<> struct IsScalar<double       > { enum { value = true }; };
template<> struct IsScalar<long double  > { enum { value = true }; };

} // namespace parex

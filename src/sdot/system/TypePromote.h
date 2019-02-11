#pragma once

#include <cstdint>

/**
*/
template<class T,class U,class Op=void>
struct TypePromote {
};


template<class T,class Op>
struct TypePromote<T,T,Op> {
    using type = T;
};

template<class Op> struct TypePromote<std::int8_t ,std::int16_t,Op> { using type = std::int16_t; };
template<class Op> struct TypePromote<std::int8_t ,std::int32_t,Op> { using type = std::int32_t; };
template<class Op> struct TypePromote<std::int8_t ,std::int64_t,Op> { using type = std::int64_t; };

template<class Op> struct TypePromote<std::int16_t,std::int32_t,Op> { using type = std::int32_t; };
template<class Op> struct TypePromote<std::int16_t,std::int64_t,Op> { using type = std::int64_t; };

template<class Op> struct TypePromote<std::int32_t,std::int64_t,Op> { using type = std::int64_t; };

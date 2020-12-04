#pragma once

#include <cstdint>
#include <string>

namespace parex {

template<class T> struct TypeInfo;

#define DECL_EXPLICIT_TYPE_INFO( NAME ) \
    template<> struct parex::TypeInfo<NAME> { static std::string name() { return #NAME; } }

DECL_EXPLICIT_TYPE_INFO( std::string   );

DECL_EXPLICIT_TYPE_INFO( double        );
DECL_EXPLICIT_TYPE_INFO( float         );

DECL_EXPLICIT_TYPE_INFO( std::int8_t   );
DECL_EXPLICIT_TYPE_INFO( std::int16_t  );
DECL_EXPLICIT_TYPE_INFO( std::int32_t  );
DECL_EXPLICIT_TYPE_INFO( std::int64_t  );

DECL_EXPLICIT_TYPE_INFO( std::uint8_t  );
DECL_EXPLICIT_TYPE_INFO( std::uint16_t );
DECL_EXPLICIT_TYPE_INFO( std::uint32_t );
DECL_EXPLICIT_TYPE_INFO( std::uint64_t );

} // namespace parex

#pragma once

#include <ostream>
#include <tuple>

namespace sdot {
namespace internal {
namespace RecursivePolytop {

template<class Cnn>
struct OrientedConnectivity {
    void                 write_to_stream( std::ostream &os ) const { os << ( neg ? "-" : "+" ) << connectivity->tmp_num; }
    bool                 operator<      ( const OrientedConnectivity &that ) const { return connectivity < that.connectivity; }
    OrientedConnectivity operator-      () const { return { connectivity, ! neg }; }

    Cnn*                 connectivity;
    bool                 neg;
};

} // namespace sdot
} // namespace internal
} // namespace RecursivePolytop

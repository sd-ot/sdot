#include "Kernel.h"
#include <tuple>

namespace parex {

bool Kernel::operator<( const Kernel &that ) const {
    return std::tie( name, /*parameters, */func ) < std::tie( that.name, /*that.parameters, */that.func );
}

} // namespace parex

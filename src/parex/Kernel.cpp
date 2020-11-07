#include "Kernel.h"
#include <tuple>

namespace parex {

bool Kernel::operator<( const Kernel &that ) const {
    return std::tie( name, modified ) < std::tie( that.name, that.modified );
}

} // namespace parex

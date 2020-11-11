#include "Kernel.h"
#include <tuple>

namespace parex {

bool Kernel::operator<( const Kernel &that ) const {
    return std::tie( name, priority, task_as_arg ) < std::tie( that.name, that.priority, that.task_as_arg );
}

} // namespace parex

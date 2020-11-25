#include "Kernel.h"

namespace parex {

bool Kernel::operator<( const Kernel &that ) const {
    return tie() < that.tie();
}

Kernel Kernel::with_vararg_num( unsigned vararg_num, std::string vararg_default_type, std::string vararg_enforced_type, const Kernel &that ) {
    Kernel res = that;
    res.vararg_num = vararg_num;
    res.vararg_default_type = vararg_default_type;
    res.vararg_enforced_type = vararg_enforced_type;
    return res;
}

} // namespace parex

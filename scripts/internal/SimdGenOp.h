#pragma once

#include <vector>
#include <string>

/**

*/
class SimdGenOp {
public:
    enum class      Type       { Undefined, Variable, Gather, Div, Mul, Add, Sub };
    using           ST         = std::size_t;

    /**/            SimdGenOp  ( Type type, ST len, std::string scalar_type );

    static bool     commutative( Type type );
    std::string     str_op     () const;

    std::string     scalar_type;
    std::vector<ST> children;
    std::vector<ST> parents;
    std::vector<ST> nouts;   /* for Gather   */
    Type            type;
    std::string     name;    /* for Variable */
    ST              len;
};

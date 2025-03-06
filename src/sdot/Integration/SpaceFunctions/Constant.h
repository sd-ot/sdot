#pragma once

#include <string>

namespace sdot {
namespace SpaceFunctions {

/**
*/
template<class TF>
class Constant {
public:
    operator bool() const { return coeff; }
    std::string name() const { return "Constant"; }

    TF coeff;
};

} // namespace SpaceFunctions
} // namespace sdot

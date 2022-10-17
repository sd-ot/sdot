#pragma once

namespace sdot {
namespace SpaceFunctions {

/**
*/
template<class TF>
class Constant {
public:
    operator bool() const { return coeff; }

    TF coeff;
};

} // namespace SpaceFunctions
} // namespace sdot

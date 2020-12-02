#pragma once

#include "ProcessingUnit.h"
#include <memory>

namespace asimd {
namespace processing_units {

/**

*/
class Discovery {
public:
    Discovery();

    std::unique_ptr<ProcessingUnit> processing_units;
};

} // namespace processing_units
} // namespace asimd

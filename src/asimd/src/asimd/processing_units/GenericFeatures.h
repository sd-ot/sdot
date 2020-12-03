#pragma once

#include <string>

namespace asimd {
namespace processing_units {
namespace features {

struct NumBoard { static std::string name() { return "NumBoard"; } std::size_t num    = 0; };
struct L1Cache  { static std::string name() { return "L1Cache" ; } std::size_t amount = 0; };
struct L2Cache  { static std::string name() { return "L2Cache" ; } std::size_t amount = 0; };

} // namespace features
} // namespace processing_units
} // namespace asimd

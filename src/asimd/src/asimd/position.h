#pragma once

namespace asimd {
namespace position {

template<int alig> struct Cpu {};

struct Gpu { int num_gpu; };

} // namespace position
} // namespace asimd

#pragma once

namespace asimd {

/**
  std::integral_constant
*/
template<int n>
struct N {
    enum { value = n };

    operator int() const { return n; }

    template<class OS>
    void write_to_stream( OS &os ) const {
        os << n;
    }

    N<-n> operator-() const {
        return {};
    }
};

} // namespace asimd

#pragma once

namespace sdot {

/**
  std::integral_constant
*/
template<int n>
struct N {
    enum { val = n };
    operator int() const { return n; }

    template<class OS>
    void write_to_stream( OS &os ) const {
        os << n;
    }

    N<-n> operator-() const {
        return {};
    }
};

} // namespace sdot

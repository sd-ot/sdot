#include "HwInfo.h"

namespace sdot {

HwInfo::HwInfo() {
}

bool sdot::HwInfo::has_AVX2() const {
    return true;
}

}

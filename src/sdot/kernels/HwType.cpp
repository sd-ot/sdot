#include "HwType.h"

//// nsmake obj_name variants/HwType_Cpu_AVX2.cpp
//// nsmake obj_name variants/HwType_Cpu_Gen.cpp

namespace sdot {

HwType *HwType::last = nullptr;

HwType::HwType( std::string TF_name, std::string TI_name ) : TF_name( TF_name ), TI_name( TI_name ), prev( last ) {
    last = this;
}

}
